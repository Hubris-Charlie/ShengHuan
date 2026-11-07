"""对话引擎：状态机 + 槽位 + 合规闸门 + 工具回写 + 全链路埋点。

一轮对话的顺序（与真实语音链路一一对应）：

    用户说完 → ASR final → 意图/槽位 → 状态转移 → 拼 prompt
    → LLM 流（逐 token 进合规闸门）→ 子句级 TTS 流 → 播放 → 埋点

这个类同时服务两种用法：
- 文本模拟（eval / 单测）：run_scripted() 跑完整一通电话
- 实时音频（ws_server）：外部按 turn() 逐轮驱动
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .flow import Intent, SlotStore, State, STATES, StateSpec, transition
from .guard import MandatoryChecker, MandatorySpec, StreamingGuard, load_banned
from .policy import HOTLINE, CallContext, CallDecision, Scenario, decide_call, get_scenario
from . import policy
from .session import EventLog, TurnTimeline, now
from .tools import MockBackend, Outbox, ToolGateway, ToolResult, validate_writeback
from . import providers as prov

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(rel: str) -> dict[str, Any]:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 意图 / 槽位

AFFIRM = ("我是", "是我", "对", "是的", "嗯", "没错", "就是", "在", "我就是", "好的", "好", "是的呢", "是")
DENY = ("不是", "打错", "没有这个", "没这个人", "不认识", "不是本人", "你打错了", "没这家", "不是我们")
ESCALATE = ("投诉", "举报", "曝光", "微博", "小红书", "媒体", "12315", "消协", "工商局", "越级", "起诉", "律师", "曝光你们", "发到网上")
OBJECTION = ("明明是真的", "明明是真", "怎么会不属实", "固定模板", "改不了", "品牌方不给", "凭什么", "不合理", "没道理",
             "已经在用", "一直都是", "骗人", "骗子", "你们搞错")
HANDOFF = ("转人工", "人工", "真人", "找个人", "客服", "找你们领导", "负责人接")
ACK = ("听明白", "听清", "明白", "知道了", "懂了", "好的", "行", "可以", "嗯", "清楚了", "记下了")

PHONE_RE = re.compile(r"1[3-9]\d{9}")
NAME_RE = re.compile(r"(?:我姓|我姓是|我叫|免贵姓)([\u4e00-\u9fa5]{1,3})")
# 商家确认身份时可能自报公司名
COMPANY_RE = re.compile(r"(?:我们是|我是|这边是)([\u4e00-\u9fa5A-Za-z0-9（）()]{2,30}(?:公司|店|商行|商贸|厂|集团))")


@dataclass
class ParseResult:
    intent: Intent
    slots: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5


class RuleNLU:
    """规则版 NLU。接真实 LLM 后替换 parse()，接口不变。

    优先级有意为之：越级投诉 > 转人工 > 异议 > 正常应答。
    因为投诉/曝光一旦说出口，就必须走升级流程，不能被"好的"这类字眼盖掉。
    """

    def parse(self, text: str, state: State) -> ParseResult:
        t = text.strip()
        slots: dict[str, Any] = {}

        m = PHONE_RE.search(t)
        if m:
            slots["phone"] = m.group(0)
        m = NAME_RE.search(t)
        if m:
            slots["customer_name"] = m.group(1)
        m = COMPANY_RE.search(t)
        if m:
            slots["merchant_name"] = m.group(1)

        if any(k in t for k in ESCALATE):
            return ParseResult(Intent.ESCALATE, slots, 0.95)
        if any(k in t for k in HANDOFF):
            return ParseResult(Intent.HANDOFF, slots, 0.9)
        if any(k in t for k in OBJECTION):
            return ParseResult(Intent.OBJECTION, slots, 0.85)

        has_affirm = any(k in t for k in AFFIRM)
        has_deny = any(k in t for k in DENY)

        if state is State.VERIFY_MERCHANT:
            if has_deny:
                slots["wrong_merchant"] = True
                return ParseResult(Intent.DENY, slots, 0.9)
            if has_affirm or slots.get("merchant_name"):
                slots["merchant_confirmed"] = True
                return ParseResult(Intent.AFFIRM, slots, 0.9)
        if state is State.REJECT_REASON and any(k in t for k in ACK):
            slots["reason_ack"] = True
        if state is State.RECTIFY and any(k in t for k in ACK):
            slots["rectify_ack"] = True

        if has_affirm:
            return ParseResult(Intent.AFFIRM, slots, 0.8)
        if has_deny:
            return ParseResult(Intent.DENY, slots, 0.8)
        if "?" in t or "？" in t or "吗" in t or "怎么" in t or "哪里" in t or "多少" in t:
            return ParseResult(Intent.QUESTION, slots, 0.7)
        return ParseResult(Intent.UNCLEAR, slots, 0.3)


# ---------------------------------------------------------------- 输出结构

@dataclass
class AgentTurn:
    turn: int
    state_before: State
    state_after: State
    reason: str
    reply: str
    guard_tripped: str | None
    used_fallback: bool
    tool: ToolResult | None
    timeline: TurnTimeline
    audio: bytes = b""          # 本轮合成的 PCM16@16k，供实时通路推给浏览器


@dataclass
class CallResult:
    session_id: str
    turns: list[AgentTurn]
    closed: bool
    final_state: State
    slots: dict[str, Any]
    writeback: dict[str, Any] | None
    transcript: list[dict[str, str]]
    decision: dict[str, Any] | None = None   # 关单方式判定结果
    called: bool = True                      # False = 判定为不该打，未拨出


# ---------------------------------------------------------------- 引擎

class Engine:
    def __init__(
        self,
        *,
        scripts: dict | None = None,
        banned_cfg: dict | None = None,
        runtime: dict | None = None,
        backend: MockBackend | None = None,
        session_id: str | None = None,
        log_dir: str | None = None,
        scenario: Any = None,
        call_ctx: Any = None,
    ) -> None:
        self.scripts = scripts or load_json("config/scripts.json")
        self.banned_cfg = banned_cfg or load_json("config/banned.json")
        self.runtime = runtime or load_json("config/runtime.json")

        self.agent_role = self.scripts.get("agent_role", "京东商家资质审核专员")
        self.agent_no = self.scripts.get("agent_no", "")
        self.hotline = self.scripts.get("hotline", HOTLINE)
        self.defaults = self.scripts.get("defaults", {})

        # 关单方式判定：先决定这次到底该不该打电话
        self.scenario = scenario or policy.get_scenario("S001")
        self.call_ctx = call_ctx or policy.CallContext()
        if self.scenario is not None:
            self.decision = policy.decide_call(self.scenario, self.call_ctx)
        else:
            self.decision = None

        banned_list, mand_specs = load_banned(self.banned_cfg)
        self.mandatory = MandatoryChecker(mand_specs)

        self.backend = backend or MockBackend()
        self.gateway = ToolGateway(self.backend)
        self.outbox = Outbox(path=os.path.join(log_dir, "outbox.json") if log_dir else None)
        self.log = EventLog(session_id, path=os.path.join(log_dir, f"{session_id or 'call'}.jsonl") if log_dir else None)
        self.session_id = self.log.session_id

        self.slots = SlotStore()
        self.state = State.OPENING
        self.return_to: State = State.REJECT_REASON
        self.nlu = RuleNLU()
        self.turn_idx = 0
        self.reprompts = 0
        self.transcript: list[dict[str, str]] = []
        self.turns: list[AgentTurn] = []
        self.writeback: dict[str, Any] | None = None
        self.guard_trips = 0
        # 话术轮转：同一状态换着说法，避免整通电话重复同一句
        self._say_idx: dict[str, int] = {}
        self._say_seed = int(hashlib.sha1(self.session_id.encode("utf-8")).hexdigest()[:4], 16)
        self._pending_closed: State | None = None

        self.llm, self.tts = self._build_providers()
        self.log.emit("system", "call_start", state=self.state.value,
                      scenario=self.scenario.id if self.scenario else None,
                      decision=self.decision.to_dict() if self.decision else None)

    # ---------- provider 装配 ----------
    def _build_providers(self) -> tuple[Any, Any]:
        sim = self.runtime.get("simulate", True)
        lat = self.runtime.get("latency_ms", {})
        scale = float(self.runtime.get("latency_scale", 1.0))

        if sim:
            ml = prov.MockLatency(
                llm_ttft_ms=tuple(lat.get("llm_ttft", [260, 520])),
                tokens_per_sec=float(lat.get("tokens_per_sec", 55)),
                tts_ttfb_ms=tuple(lat.get("tts_ttfb", [140, 260])),
                tts_char_ms=float(lat.get("tts_char_ms", 165)),
                scale=scale,
            )
            llm = prov.MockLLM(ml)
            tts = prov.MockTTS(tuple(lat.get("tts_ttfb", [140, 260])), float(lat.get("tts_char_ms", 165)), scale=scale)
            return llm, tts

        p = self.runtime.get("providers", {})
        llm_cfg = p.get("llm", {})
        llm = prov.OpenAICompatLLM(llm_cfg.get("base_url", ""), llm_cfg.get("model", ""))
        tts_cfg = p.get("tts", {})
        tts = prov.CloudTTSStub(tts_cfg.get("vendor", ""), tts_cfg.get("api_key_env", ""), tts_cfg.get("voice", ""), tts_cfg.get("model", ""))
        return llm, tts

    # ---------- 文本渲染 ----------
    def render(self, template: str) -> str:
        """填充占位符。支持 {key} 与【key】两种写法（原始话术表用的是【】）。"""
        scen = self.scenario
        ctx: dict[str, str] = {
            "role": self.agent_role,
            "工号": self.agent_no,
            "hotline": self.hotline,
            "热线": self.hotline,
            "品牌": (self.slots.value("brand") or (scen.script and "") or self.defaults.get("品牌", "XX")),
            "类目": self.slots.value("category") or self.defaults.get("类目", "XXXX"),
            "驳回原因": self.slots.value("reject_reason") or self._reason_text(),
            "整改动作": self.slots.value("rectify_action") or self._rectify_text(),
            "商家": self.slots.value("merchant_name") or self.defaults.get("商家", "（商家公司名/店铺名称）"),
            "称呼": self.slots.value("customer_name") or "您",
        }
        out = template
        for k, v in ctx.items():
            out = out.replace("{" + k + "}", str(v)).replace("【" + k + "】", str(v))
        # 兜底：把没填上的【】占位符换成通用说法，避免把方括号念出来
        out = re.sub(r"【([^】]{0,8})】", lambda m: self.defaults.get(m.group(1), m.group(1) or "相关"), out)
        return out

    def _reason_text(self) -> str:
        """驳回原因的播报文本：按场景二级分类清洗成能直接念的口语。"""
        if self.scenario is not None:
            return policy.reason_text(self.scenario)
        return self.defaults.get("驳回原因", "提交的资质材料不符合申请标准")

    def _rectify_text(self) -> str:
        """整改动作：优先用官方话术里能干净拆出的那句，否则用一级分类兜底。"""
        override = self.slots.value("rectify_action")
        if override:
            return str(override)
        opts = policy.action_options(self.scenario)
        if not opts:
            return self.defaults.get("整改动作", "补齐对应资质材料")
        key = "action"
        i = self._say_idx.get(key, self._say_seed % len(opts))
        self._say_idx[key] = i + 1
        return opts[i % len(opts)]

    def _rectify_bucket(self) -> str:
        """按整改动作的句子形态选模板，避免出现"请您暂无自助修改方式…"这种病句。"""
        a = self._rectify_text()
        if a.startswith(("暂无", "这个场景")):
            return "as_is"
        if any(k in a for k in ("提交", "申请", "上传")):
            return "self_contained"
        return "needs_submit"

    # ---------- 话术选取 ----------
    def _pick(self, bucket: str, state: State) -> str | None:
        """从某个话术桶里按轮转取一句，保证同一状态不会连续重复同一句话。

        bucket: "utterances" | "reprompts"
        起点按会话种子错开，让不同通话的开场白不一样。
        """
        items = self.scripts.get(bucket, {}).get(state.value) or []
        if not items:
            return None
        key = f"{bucket}:{state.value}"
        idx = self._say_idx.get(key)
        if idx is None:
            # 用键名散列 + 会话种子决定起点，避免每次通话都从第一句开始
            idx = (self._say_seed + sum(ord(c) for c in key)) % len(items)
        self._say_idx[key] = idx + 1
        return self.render(items[idx % len(items)])

    def _pick_any(self, values: Any) -> str:
        """兜底话术既支持单句字符串，也支持多句数组。"""
        if isinstance(values, str):
            return self.render(values)
        items = list(values or [])
        if not items:
            return ""
        key = "fallback:" + (items[0][:8])
        i = self._say_idx.get(key, 0) % len(items)
        self._say_idx[key] = i + 1
        return self.render(items[i])

    def _utterance(self, state: State) -> str:
        # 整改动作是按场景动态变的，用专门的模板桶，而不是固定的 utterances
        if state is State.RECTIFY:
            items = (self.scripts.get("rectify_templates") or {}).get(self._rectify_bucket()) or []
            if items:
                key = f"rectify:{self._rectify_bucket()}"
                i = self._say_idx.get(key, self._say_seed % len(items))
                self._say_idx[key] = i + 1
                return self.render(items[i % len(items)])
        said = self._pick("utterances", state)
        if said:
            return said
        fb = self.scripts.get("fallbacks", {})
        return self._pick_any(fb.get("generic_fallback", ""))

    def _reprompt(self, state: State) -> str:
        """没听清 / 没拿到槽位时的追问话术。没有专用追问就用兜底。"""
        said = self._pick("reprompts", state)
        if said:
            return said
        return self._fallback(STATES[state])

    def _fallback(self, spec: StateSpec) -> str:
        fb = self.scripts.get("fallbacks", {})
        val = fb.get(spec.fallback) or fb.get("generic_fallback", "")
        out = self._pick_any(val)
        return out or "不好意思，我没听清，您能再说一遍吗？"

    def build_prompt(self, spec: StateSpec, user_text: str) -> list[dict[str, str]]:
        """真实 LLM 用的 prompt。约束都写在 system 里，不靠模型自觉。"""
        slot_desc = json.dumps(self.slots.flat(), ensure_ascii=False)
        scen = self.scenario
        scen_desc = (
            f"场景：{scen.label}（关单方式：{scen.close_method.value}）\n" if scen else ""
        )
        sys = (
            f"你是{self.agent_role}，正在外呼告知商家资质申请被驳回。\n"
            f"{scen_desc}"
            f"当前通话阶段：{spec.name.value}。本阶段唯一目标：{spec.goal}\n"
            f"已收集信息：{slot_desc}\n"
            "铁律：\n"
            "1. 只做当前阶段的事，不要跳步。\n"
            "2. 未经对方确认身份，绝不讲述驳回原因。\n"
            "3. 不得承诺审核结果（不能说“补上就能过”“一定通过”）。\n"
            "4. 不得索要身份证号、银行卡号、密码。\n"
            "5. 不报联系人号码全号，只能给尾号后四位。\n"
            "6. 第一句不超过 12 个字，先短回应再说内容。\n"
            "7. 输出纯口语，不含表情、括号动作、markdown。\n"
        )
        hist = self.transcript[-6:]
        msgs = [{"role": "system", "content": sys}]
        for h in hist:
            msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": user_text or "（接通）"})
        return msgs

    # ---------- 一轮 ----------
    async def opening(self) -> AgentTurn:
        return await self._speak(turn=0, state_before=State.OPENING, state_after=State.VERIFY_MERCHANT,
                                 reason="opening_done", text=self._utterance(State.OPENING),
                                 user_text="")

    async def turn(self, user_text: str) -> AgentTurn:
        self.turn_idx += 1
        tl = TurnTimeline(turn=self.turn_idx)
        # 文本模式下没有真实声学端点，按配置的静音窗还原一个可对齐的值
        end_ms = float(self.runtime.get("limits", {}).get("endpoint_silence_ms", 280))
        tl.t_speech_end = self.log.t()
        tl.t_speech_start = tl.t_speech_end - end_ms / 1000.0
        self.log.emit("asr", "final", turn=self.turn_idx, text=user_text)

        spec = STATES[self.state]
        parsed = self.nlu.parse(user_text, self.state)
        self.log.emit("nlu", "parse", turn=self.turn_idx, intent=parsed.intent.value,
                      slots=parsed.slots, conf=parsed.confidence)

        # 槽位写入（改口走确认-覆盖）
        for k, v in parsed.slots.items():
            act = self.slots.set(k, v, parsed.confidence, user_text)
            self.log.emit("slot", act, turn=self.turn_idx, name=k, value=v)

        tr = transition(spec, self.slots, parsed.intent, reprompt_count=self.reprompts)
        # 异议处理后要回到原流程，而不是固定去 NEED
        if spec.name is State.OBJECTION and tr.to_state is not State.HANDOFF:
            tr = type(tr)(spec.name, self.return_to, "objection_handled")
        if spec.name is not State.OBJECTION and tr.to_state is State.OBJECTION:
            self.return_to = spec.name

        self.log.emit("flow", "transition", turn=self.turn_idx,
                      frm=tr.from_state.value, to=tr.to_state.value, reason=tr.reason)

        # 选回复文本
        used_fallback = False
        if tr.to_state is spec.name:  # 还在原地 = 需要重问
            self.reprompts += 1
            text = self._reprompt(spec.name) if self.reprompts <= 1 else self._fallback(spec)
        elif tr.reason == "wrong_merchant":
            # 打错人：说专用道歉话术，不能念常规结束语
            text = self._pick_any((self.scripts.get("special") or {}).get("wrong_merchant_close") or self._fallback(spec))
        elif tr.reason == "unverified_close":
            # 身份确认不了：说结束语挂断，不能还在这句上反复追问
            text = self._pick_any((self.scripts.get("special") or {}).get("unverified_close") or self._fallback(spec))
        elif tr.reason.startswith("forced_advance"):
            used_fallback = True
            text = self._fallback(spec)
        elif tr.to_state is State.OBJECTION:
            text = self._utterance(State.OBJECTION)
        elif tr.to_state is State.HANDOFF:
            text = self._utterance(State.HANDOFF)
        elif tr.to_state is State.ESCALATE:
            text = self._utterance(State.ESCALATE)
        else:
            self.reprompts = 0
            text = self._utterance(tr.to_state) if tr.to_state in STATES else self._utterance(spec.name)

        # 进入结束类状态（礼貌结束 / 升级）时做工具调用与回写
        # 注意：_speak 会把 self.state 推到新状态，所以回写要在这里先做
        tool: ToolResult | None = None
        if tr.to_state in (State.CLOSE, State.ESCALATE):
            self._pending_closed = tr.to_state
            tool = self._do_writeback()

        return await self._speak(turn=self.turn_idx, state_before=spec.name, state_after=tr.to_state,
                                 reason=tr.reason, text=text, user_text=user_text,
                                 timeline=tl, used_fallback=used_fallback, tool=tool)

    # ---------- 工具与回写 ----------
    def _do_writeback(self) -> ToolResult:
        """回写"已外呼告知驳回原因"。

        业务前提：**没确认对方身份就不能回写已告知**。给错人讲驳回原因是重大事故，
        所以 merchant_confirmed 是硬门槛，不是可选项。
        """
        scen = self.scenario
        confirmed = self.slots.value("merchant_confirmed") is True
        reason_ack = self.slots.value("reason_ack") is True
        rectify_ack = self.slots.value("rectify_ack") is True
        escalation = (
            self.call_ctx.escalation_risk
            or self.state is State.ESCALATE
            or self._pending_closed is State.ESCALATE
        )

        if not confirmed:
            call_result = "wrong_number" if self.slots.value("wrong_merchant") is True else "unverified"
        elif escalation:
            call_result = "escalated"
        elif reason_ack:
            call_result = "notified"
        else:
            call_result = "partial"

        payload = {
            "session_id": self.session_id,
            "scenario_id": scen.id if scen else "",
            "menu1": scen.menu1 if scen else "",
            "cat1": scen.cat1 if scen else "",
            "cat2": scen.cat2 if scen else "",
            "close_method": self.decision.method.value if self.decision else "",
            "merchant_confirmed": confirmed,
            "reason_ack": reason_ack,
            "rectify_ack": rectify_ack,
            "escalation_risk": bool(escalation),
            "call_result": call_result,
        }
        ok, missing = validate_writeback(payload)
        if not ok:
            # 身份没确认 → 不建单、不回写"已告知"。宁可不记，也不能记错。
            self.log.emit("tool", "writeback_skipped", reason=missing, payload=payload)
            return ToolResult(True, "create_ticket", data={}, created=False, idem_key="")

        res = self.gateway.call(self.session_id, "create_ticket", {
            "session_id": self.session_id,
            "scenario_id": payload["scenario_id"],
            "call_result": call_result,
        })
        if res.ok:
            payload["ticket_id"] = res.data.get("ticket_id", "")
        payload["_idem_key"] = res.idem_key
        rec = self.outbox.submit(payload)
        self.outbox.dispatch(lambda p, k: True)
        self.log.emit("tool", "outbox", status=rec["status"], idem=rec["idem_key"], payload=payload)
        self.writeback = payload
        return res

    # ---------- 发声（含埋点 + 合规闸门） ----------
    async def _speak(
        self,
        *,
        turn: int,
        state_before: State,
        state_after: State,
        reason: str,
        text: str,
        user_text: str,
        timeline: TurnTimeline | None = None,
        used_fallback: bool = False,
        tool: ToolResult | None = None,
    ) -> AgentTurn:
        tl = timeline or TurnTimeline(turn=turn)
        if tl.t_speech_end is None:
            tl.t_speech_end = self.log.t()
        tl.t_asr_final = self.log.t()

        guard = StreamingGuard(self.banned_cfg.get("banned", []))
        released_parts: list[str] = []
        clause_buf = ""
        total_dur = 0.0
        tripped: str | None = None
        gen_chars: list[str] = []
        self._audio_acc = []

        tl.t_llm_req = self.log.t()
        async for tok in self.llm.stream(self.build_prompt(STATES[self.state], user_text), mock_reply=text):
            if tl.t_llm_first_token is None:
                tl.t_llm_first_token = self.log.t()
                self.log.emit("llm", "first_token", turn=turn)
            gen_chars.append(tok)
            safe, hit = guard.feed(tok)
            if hit:
                tripped = hit.detail
                self.guard_trips += 1
                self.log.emit("guard", "banned", turn=turn, pattern=hit.detail, context=hit.text)
                break
            if safe:
                released_parts.append(safe)
                clause_buf += safe
                # 子句级流式合成：不等整段
                while True:
                    idx = _clause_cut(clause_buf)
                    if idx <= 0:
                        break
                    clause, clause_buf = clause_buf[:idx], clause_buf[idx:]
                    first, dur = await self._synthesize(clause, tl, turn)
                    total_dur += dur

        if tripped is None:
            tail, hit2 = guard.flush()
            if hit2:
                tripped = hit2.detail
                self.guard_trips += 1
                self.log.emit("guard", "banned_flush", turn=turn, pattern=hit2.detail)
            elif tail:
                released_parts.append(tail)
                clause_buf += tail
                first, dur = await self._synthesize(clause_buf, tl, turn)
                total_dur += dur

        used_fallback = used_fallback or tripped is not None
        if tripped is not None:
            # 命中禁语：整段丢弃，改说兜底话术（确定性文本，必然合规）
            spec = STATES[self.state]
            safe_text = self._fallback(spec)
            released_parts = []
            total_dur = 0.0
            for clause in prov._split_clause(safe_text):
                first, dur = await self._synthesize(clause, tl, turn)
                total_dur += dur
            reply = safe_text
        else:
            reply = "".join(released_parts)

        if tl.t_playout_start is None:
            tl.t_playout_start = self.log.t()
        tl.t_playout_end = tl.t_playout_start + total_dur / 1000.0

        self.mandatory.observe(reply)
        self.mandatory.visit(state_before.value)
        self.mandatory.visit(state_after.value)
        self.transcript.append({"role": "assistant", "content": reply})
        if user_text:
            self.transcript.append({"role": "user", "content": user_text})

        self.log.emit("tts", "spoken", turn=turn, chars=len(reply), dur_ms=round(total_dur, 1),
                      text=reply, tripped=tripped, fallback=used_fallback)
        self.log.emit("turn", "done", turn=turn, timeline=tl.to_dict())

        at = AgentTurn(
            turn=turn,
            state_before=state_before,
            state_after=state_after,
            reason=reason,
            reply=reply,
            guard_tripped=tripped,
            used_fallback=used_fallback,
            tool=tool,
            timeline=tl,
            audio=b"".join(self._audio_acc),
        )
        # 状态推进放在最后：保证本轮读的是推进前的 spec
        self.state = state_after
        self.turns.append(at)
        return at

    async def _synthesize(self, clause: str, tl: TurnTimeline, turn: int) -> tuple[bool, float]:
        dur = prov.estimate_duration_ms(clause, self.runtime.get("latency_ms", {}).get("tts_char_ms", 165))
        first = False
        async for audio in self.tts.stream(clause):
            if audio:
                self._audio_acc.append(audio)
            if tl.t_tts_first_byte is None:
                tl.t_tts_first_byte = self.log.t()
                self.log.emit("tts", "first_byte", turn=turn)
                first = True
            if tl.t_playout_start is None and len(clause) > 0:
                tl.t_playout_start = tl.t_tts_first_byte
        return first, dur

    # ---------- 整通电话 ----------
    async def run_scripted(self, user_lines: list[str], max_turns: int | None = None) -> CallResult:
        limit = max_turns or int(self.runtime.get("limits", {}).get("max_turns", 12))

        # 关单方式判定为"不该打"时，一通电话都不拨
        if self.decision is not None and not self.decision.should_call:
            self.log.emit("policy", "call_skipped", decision=self.decision.to_dict())
            self.log.emit("system", "call_end", state="NOT_CALLED", turns=0)
            return CallResult(
                session_id=self.session_id,
                turns=[],
                closed=False,
                final_state=self.state,
                slots=self.slots.flat(),
                writeback=None,
                transcript=[],
                decision=self.decision.to_dict(),
                called=False,
            )

        await self.opening()
        for line in user_lines:
            if self.state in (State.CLOSED, State.CLOSE, State.ESCALATE, State.HANDOFF, State.EXCEPTION):
                break
            if len(self.turns) >= limit:
                break
            await self.turn(line)
        closed = self.state in (State.CLOSED, State.CLOSE)
        if not closed and self.state not in (State.ESCALATE, State.HANDOFF):
            self.log.emit("flow", "unclosed", state=self.state.value)
        self.log.emit("system", "call_end", state=self.state.value, turns=len(self.turns))
        return CallResult(
            session_id=self.session_id,
            turns=self.turns,
            closed=closed or self.state in (State.ESCALATE, State.HANDOFF),
            final_state=self.state,
            slots=self.slots.flat(),
            writeback=self.writeback,
            transcript=self.transcript,
            decision=self.decision.to_dict() if self.decision else None,
            called=True,
        )

    def compliance(self) -> dict[str, Any]:
        cov = self.mandatory.coverage()
        return {
            "mandatory_coverage": cov,
            "mandatory_missing": [s.uid for s in self.mandatory.missing()],
            "guard_trips": self.guard_trips,
        }


def _clause_cut(buf: str) -> int:
    """返回一个子句的切点（含标点），没有则返回 0。

    目的：让 TTS 首包不必等整段生成完 —— 攒够一个子句就先合成。
    """
    best = 0
    for i, ch in enumerate(buf):
        if ch in "，。？！、；：,.?!":
            if i + 1 >= 3:      # 太短就再攒一点，避免碎句
                return i + 1
        elif i >= 12:           # 没有标点也要切，防止长句卡住首包
            best = i + 1
            break
    return best
