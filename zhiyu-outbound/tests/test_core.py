"""核心逻辑单测（资质审核驳回外呼场景）。

覆盖：
1. 合规闸门（跨 chunk 禁语、归一化变体）
2. 状态机与槽位（身份确认硬闸门、追问耗尽、改口确认、升级/转人工）
3. 打断的播放进度记账
4. 工具幂等 + 回写校验 + 整通电话闭环
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import policy
from server.flow import Intent, SlotStore, State, STATES, transition
from server.guard import MandatoryChecker, MandatorySpec, StreamingGuard, normalize
from server.session import TurnTimeline, build_report, percentile
from server.tools import MockBackend, Outbox, ToolGateway, make_idem_key, validate_writeback
from server.turn import EchoGate, TurnController


def phone_scene():
    return policy.find_scenario("品牌", "商标注册证", "商标状态不符合要求")


def jingmai_scene():
    return policy.find_scenario("品牌", "商标注册证", "资质造假（公章造假/链路造假等）")


class TestGuard(unittest.TestCase):
    def test_normalize(self):
        # 中文数字会被归一化成阿拉伯数字。禁语表与文本两侧都走同一归一化，
        # 所以匹配不受影响；但报告里用 _orig 还原成原文。
        self.assertEqual(normalize("一定 通过！"), "1定通过")
        self.assertEqual(normalize("４００６１８００９２"), "4006180092")

    def test_banned_plain(self):
        g = StreamingGuard(["一定通过"])
        out, hit = g.feed("这个一定通过没问题的")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.detail, "一定通过")
        self.assertEqual(out, "")

    def test_banned_split_across_chunks(self):
        """禁语跨 chunk 必须能被抓到 —— 流式拦截的核心难点。"""
        g = StreamingGuard(["包您通过"])
        hit_any = None
        for ch in "这个我包您通过":
            _, hit = g.feed(ch)
            if hit:
                hit_any = hit
                break
        self.assertIsNotNone(hit_any, "跨 chunk 的禁语没被拦到")
        self.assertEqual(hit_any.detail, "包您通过")

    def test_safe_text_released_after_flush(self):
        g = StreamingGuard(["一定通过"])
        text = "您申请的商标状态不符合要求，需要您补齐后重新提交。"
        got = ""
        for ch in text:
            safe, hit = g.feed(ch)
            self.assertIsNone(hit)
            got += safe
        tail, hit = g.flush()
        self.assertIsNone(hit)
        got += tail
        self.assertEqual(got, text, "扣留窗口把正常文本吞掉了")

    def test_mandatory_coverage(self):
        m = MandatoryChecker([MandatorySpec("hotline", "热线", [["4006180092"]])])
        m.observe("如有疑问您可以电话联系：4006180092，再见")
        self.assertTrue(m.coverage()["hotline"])

    def test_mandatory_conditional(self):
        spec = MandatorySpec("escalation", "升级", [["记录"]], requires_state="ESCALATE")
        m = MandatoryChecker([spec])
        m.observe("您好")
        self.assertEqual(m.missing(), [])
        m.visit("ESCALATE")
        self.assertEqual(len(m.missing()), 1)


class TestFlow(unittest.TestCase):
    def test_reprompt_then_force_advance(self):
        slots = SlotStore()
        spec = STATES[State.REJECT_REASON]
        t1 = transition(spec, slots, Intent.UNCLEAR, reprompt_count=0)
        self.assertEqual(t1.to_state, State.REJECT_REASON)
        t2 = transition(spec, slots, Intent.UNCLEAR, reprompt_count=spec.max_reprompt)
        self.assertEqual(t2.to_state, State.RECTIFY)
        self.assertTrue(t2.reason.startswith("forced_advance"))

    def test_slots_complete_advances(self):
        slots = SlotStore()
        slots.set("reason_ack", True)
        t = transition(STATES[State.REJECT_REASON], slots, Intent.AFFIRM)
        self.assertEqual(t.to_state, State.RECTIFY)

    def test_verify_gate_blocks_unconfirmed(self):
        """身份没确认，追问用尽也不能讲驳回原因。"""
        slots = SlotStore()
        spec = STATES[State.VERIFY_MERCHANT]
        for n in range(spec.max_reprompt):
            t = transition(spec, slots, Intent.UNCLEAR, reprompt_count=n)
            self.assertEqual(t.to_state, State.VERIFY_MERCHANT)
        t = transition(spec, slots, Intent.UNCLEAR, reprompt_count=spec.max_reprompt)
        self.assertEqual(t.to_state, State.CLOSE, "未确认身份却继续往下走了")
        self.assertEqual(t.reason, "unverified_close")

    def test_verify_deny_hangs_up(self):
        t = transition(STATES[State.VERIFY_MERCHANT], SlotStore(), Intent.DENY)
        self.assertEqual(t.to_state, State.CLOSE)
        self.assertEqual(t.reason, "wrong_merchant")

    def test_verify_confirm_proceeds(self):
        slots = SlotStore()
        slots.set("merchant_confirmed", True)
        t = transition(STATES[State.VERIFY_MERCHANT], slots, Intent.AFFIRM)
        self.assertEqual(t.to_state, State.REJECT_REASON)

    def test_escalate_beats_everything(self):
        for st in (State.OPENING, State.VERIFY_MERCHANT, State.REJECT_REASON, State.RECTIFY):
            t = transition(STATES[st], SlotStore(), Intent.ESCALATE)
            self.assertEqual(t.to_state, State.ESCALATE, f"{st} 未转升级")

    def test_handoff_intent(self):
        t = transition(STATES[State.REJECT_REASON], SlotStore(), Intent.HANDOFF)
        self.assertEqual(t.to_state, State.HANDOFF)

    def test_objection_at_verify(self):
        t = transition(STATES[State.VERIFY_MERCHANT], SlotStore(), Intent.OBJECTION)
        self.assertEqual(t.to_state, State.OBJECTION)

    def test_slot_overwrite_needs_confirm(self):
        s = SlotStore()
        self.assertEqual(s.set("need", "a"), "set")
        self.assertEqual(s.set("need", "b"), "confirm_required")
        self.assertEqual(s.value("need"), "a")
        self.assertEqual(s.set("need", "b"), "set")
        self.assertEqual(s.value("need"), "b")


class TestBargeIn(unittest.TestCase):
    def test_playout_based_prefix(self):
        tc = TurnController()
        tc.on_tts_chunk("您好，我", 1000)
        tc.on_tts_chunk("是京东服务", 1000)
        tc.on_tts_chunk("的助理。", 1000)
        tc.on_gen_done()
        self.assertEqual(tc.generated_total_ms, 3000)

        tc.on_playout(1500)
        self.assertEqual(tc.spoken_prefix(), "您好，我是京")
        tc.on_playout(2000)
        self.assertEqual(tc.spoken_prefix(), "您好，我是京东服务")

    def test_barge_in_reports_unspoken(self):
        tc = TurnController()
        tc.on_tts_chunk("第一句。", 800)
        tc.on_tts_chunk("第二句。", 800)
        tc.on_playout(700)
        res = tc.barge_in()
        self.assertTrue(res.triggered)
        self.assertEqual(res.played_ms, 700)
        self.assertNotIn("第二句", res.spoken_prefix)
        self.assertIn("第二句", res.unspoken)

    def test_echo_gate_blocks_self_echo(self):
        g = EchoGate()
        g.set_agent_speaking(True, onset_ms=0)
        ok, why = g.should_trigger(now_ms=500, speech_ms=400, echo_score=0.95)
        self.assertFalse(ok)
        self.assertIn("self_echo", why)

    def test_echo_gate_allows_real_barge_in(self):
        g = EchoGate()
        g.set_agent_speaking(True, onset_ms=0)
        ok, why = g.should_trigger(now_ms=1200, speech_ms=420, echo_score=0.2)
        self.assertTrue(ok)
        self.assertEqual(why, "ok")


class TestTools(unittest.TestCase):
    def test_idempotent_ticket(self):
        be = MockBackend()
        gw = ToolGateway(be)
        args = {"session_id": "s1", "scenario_id": "S001", "call_result": "notified"}
        r1 = gw.call("s1", "create_ticket", args)
        r2 = gw.call("s1", "create_ticket", args)
        self.assertTrue(r1.ok and r2.ok)
        self.assertTrue(r1.created)
        self.assertFalse(r2.created, "重复调用不应该新建工单")
        self.assertEqual(r1.data["ticket_id"], r2.data["ticket_id"])

    def test_retry_on_injected_failure(self):
        be = MockBackend()
        be.inject_failure("query_order", 1)
        gw = ToolGateway(be, max_attempts=3)
        r = gw.call("s1", "query_order", {"order_id": "O1"})
        self.assertTrue(r.ok)
        self.assertEqual(r.attempts, 2)

    def test_idem_key_stable(self):
        a = make_idem_key("s1", "t", {"x": 1, "y": 2})
        b = make_idem_key("s1", "t", {"y": 2, "x": 1})
        self.assertEqual(a, b)

    def test_writeback_requires_identity(self):
        """没确认身份 → 不能回写"已告知"。"""
        ok, missing = validate_writeback({
            "session_id": "s1", "scenario_id": "S001", "menu1": "品牌", "cat1": "商标注册证",
            "cat2": "商标状态不符合要求", "close_method": "电话", "call_result": "notified",
            "merchant_confirmed": False,
        })
        self.assertFalse(ok)
        self.assertIn("merchant_confirmed", missing)

    def test_writeback_allows_wrong_number(self):
        """打错人是单独结局，允许在未确认身份时回写。"""
        ok, missing = validate_writeback({
            "session_id": "s1", "scenario_id": "S001", "menu1": "品牌", "cat1": "商标注册证",
            "cat2": "商标状态不符合要求", "close_method": "电话", "call_result": "wrong_number",
            "merchant_confirmed": False,
        })
        self.assertTrue(ok, missing)

    def test_writeback_happy_path(self):
        ok, missing = validate_writeback({
            "session_id": "s1", "scenario_id": "S001", "menu1": "品牌", "cat1": "商标注册证",
            "cat2": "商标状态不符合要求", "close_method": "电话", "call_result": "notified",
            "merchant_confirmed": True,
        })
        self.assertTrue(ok, missing)

    def test_outbox_rejects_invalid(self):
        ob = Outbox()
        ob.submit({"session_id": "s1"})
        sent = ob.dispatch(lambda p, k: True)
        self.assertEqual(sent[0]["status"], "rejected")


class TestSession(unittest.TestCase):
    def test_percentile(self):
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(percentile([10], 95), 10)

    def test_timeline_main_metric(self):
        tl = TurnTimeline(turn=1, t_speech_end=2.0, t_playout_start=3.1)
        self.assertAlmostEqual(tl.first_response_ms(), 1100.0)

    def test_report_build(self):
        tls = [TurnTimeline(turn=i, t_speech_end=0.0, t_playout_start=1.0 + i * 0.1) for i in range(3)]
        rep = build_report(tls)
        self.assertEqual(rep["turns"], 3)


class TestEngineIntegration(unittest.IsolatedAsyncioTestCase):
    def rt(self):
        from server.engine import load_json

        r = load_json("config/runtime.json")
        r["latency_scale"] = 0.0
        r["simulate"] = True
        return r

    async def test_full_call_closes_and_writes_back(self):
        from server.engine import Engine

        eng = Engine(runtime=self.rt(), scenario=phone_scene())
        res = await eng.run_scripted(["对，是我们", "我听明白了", "好的谢谢"])
        self.assertTrue(res.called)
        self.assertTrue(res.closed, f"没闭环，停在 {res.final_state}")
        self.assertIsNotNone(res.writeback, "没有产生回写")
        self.assertEqual(res.writeback["call_result"], "notified")
        self.assertTrue(res.writeback["ticket_id"].startswith("T"))
        self.assertEqual(res.writeback["scenario_id"], "S001")
        self.assertEqual(eng.compliance()["mandatory_missing"], [])
        self.assertEqual(eng.compliance()["guard_trips"], 0)
        for t in res.turns:
            self.assertIsNotNone(t.timeline.first_response_ms(), f"turn {t.turn} 缺时延埋点")

    async def test_jingmai_scene_not_called(self):
        from server.engine import Engine

        eng = Engine(runtime=self.rt(), scenario=jingmai_scene())
        res = await eng.run_scripted(["喂？"])
        self.assertFalse(res.called, "京麦驳回场景不该外呼")
        self.assertEqual(res.decision["method"], "京麦驳回")
        self.assertEqual(res.turns, [])

    async def test_wrong_merchant_hangs_up(self):
        from server.engine import Engine

        eng = Engine(runtime=self.rt(), scenario=phone_scene())
        res = await eng.run_scripted(["你打错了，不是我们"])
        self.assertEqual(res.final_state, State.CLOSE)
        self.assertEqual(res.writeback["call_result"], "wrong_number")
        # 绝不能对被拒的号码讲驳回原因
        self.assertTrue(all(t.state_after is not State.REJECT_REASON for t in res.turns))
        self.assertTrue(res.closed)

    async def test_escalation_mid_call(self):
        from server.engine import Engine

        eng = Engine(runtime=self.rt(), scenario=phone_scene())
        res = await eng.run_scripted(["是这里", "我要去微博曝光你们"])
        self.assertEqual(res.final_state, State.ESCALATE)
        self.assertNotIn("escalation_notice", eng.compliance()["mandatory_missing"])

    async def test_merchant_requested_callback_overrides_jingmai(self):
        from server.engine import Engine

        eng = Engine(runtime=self.rt(), scenario=jingmai_scene(),
                     call_ctx=policy.CallContext(merchant_requested_callback=True))
        res = await eng.run_scripted(["是这里"])
        self.assertTrue(res.called, "商家要求回电，即使京麦场景也必须打")
        self.assertEqual(res.decision["method"], "电话")

    async def test_guard_intercepts_bad_reply(self):
        from server.engine import Engine

        eng = Engine(runtime=self.rt(), scenario=phone_scene())
        # 把要说的那句换成会命中禁语的内容（"保证"、"一定通过"都在禁语表里）
        eng.scripts["rectify_templates"]["needs_submit"] = ["我保证这个一定通过，您放心。"]
        eng.slots.set("merchant_confirmed", True)
        eng.slots.set("reason_ack", True)
        eng.state = State.REJECT_REASON
        t = await eng.turn("嗯")
        self.assertEqual(t.state_after, State.RECTIFY)
        self.assertIsNotNone(t.guard_tripped, "禁语没被拦住")
        self.assertTrue(t.used_fallback)
        self.assertNotIn("一定通过", t.reply)
        self.assertNotIn("保证", t.reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)
