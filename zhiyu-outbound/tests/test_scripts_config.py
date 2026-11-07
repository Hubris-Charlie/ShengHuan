"""话术配置自检。

话术是**轮转**使用的，只要有一句踩禁语或缺必说话术，就说明"抽到那句就违规"，
所以必须逐句校验，不能只验第一句。

另一个关键点：校验前必须先把占位符渲染掉。
"您好，我是{role}，请问您是【商家】吗？" 里的 {role} 渲染后才是"京东商家资质审核专员"，
直接拿模板去匹配必说话术会误判。
"""
from __future__ import annotations

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.engine import Engine, load_json
from server.flow import STATES, State
from server.guard import MandatorySpec, StreamingGuard, normalize

SCRIPTS = load_json("config/scripts.json")
BANNED = load_json("config/banned.json")

ALLOWED_PLACEHOLDERS = {
    "role", "工号", "hotline", "热线",
    "品牌", "类目", "驳回原因", "整改动作", "商家", "称呼",
}
PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")
BRACKET_RE = re.compile(r"【([^】]{0,8})】")


def render(tpl: str) -> str:
    """按引擎同样的规则渲染占位符，供校验使用。"""
    ctx = {
        "role": SCRIPTS.get("agent_role", ""),
        "工号": SCRIPTS.get("agent_no", ""),
        "hotline": SCRIPTS.get("hotline", ""),
        "热线": SCRIPTS.get("hotline", ""),
    }
    ctx.update(SCRIPTS.get("defaults", {}))
    out = tpl
    for k, v in ctx.items():
        out = out.replace("{" + k + "}", str(v)).replace("【" + k + "】", str(v))
    out = BRACKET_RE.sub(lambda m: SCRIPTS.get("defaults", {}).get(m.group(1), m.group(1) or "相关"), out)
    return out


def iter_utterances():
    """产出 (bucket, level_key, index, 原始模板)。"""
    for bucket in ("utterances", "reprompts"):
        for key, items in SCRIPTS.get(bucket, {}).items():
            if key == "_note":
                continue
            for i, tpl in enumerate(items):
                yield bucket, key, i, tpl
    for key, val in SCRIPTS.get("fallbacks", {}).items():
        items = [val] if isinstance(val, str) else list(val)
        for i, tpl in enumerate(items):
            yield "fallbacks", key, i, tpl
    for bucket, items in (SCRIPTS.get("rectify_templates") or {}).items():
        if bucket == "_note":
            continue
        for i, tpl in enumerate(items):
            yield "rectify_templates", bucket, i, tpl
    for key, val in (SCRIPTS.get("filler") or {}).items():
        items = [val] if isinstance(val, str) else list(val)
        for i, tpl in enumerate(items):
            yield "filler", key, i, tpl
    for key, val in (SCRIPTS.get("special") or {}).items():
        if key == "_note":
            continue
        items = [val] if isinstance(val, str) else list(val)
        for i, tpl in enumerate(items):
            yield "special", key, i, tpl


class TestScriptsConfig(unittest.TestCase):
    def test_placeholders_are_known(self):
        """占位符写错会原样播报出来。"""
        for bucket, name, i, tpl in iter_utterances():
            for ph in PLACEHOLDER_RE.findall(tpl):
                self.assertIn(ph, ALLOWED_PLACEHOLDERS,
                              f"{bucket}/{name}[{i}] 有未知占位符 {{{ph}}}：{tpl}")

    def test_no_banned_phrase(self):
        """任何一句（渲染后）都不得命中禁语。"""
        banned = BANNED["banned"]
        for bucket, name, i, tpl in iter_utterances():
            text = render(tpl)
            guard = StreamingGuard(banned)
            _, hit = guard.feed(text)
            if hit is None:
                _, hit = guard.flush()
            self.assertIsNone(hit, f"{bucket}/{name}[{i}] 命中禁语「{hit.detail if hit else ''}」：{text}")

    def test_mandatory_satisfied_by_every_variant(self):
        """声明了必说话术的状态，名下每一句变体（渲染后）都必须满足。"""
        specs = {m["uid"]: MandatorySpec(uid=m["uid"], label=m.get("label", ""), groups=m["groups"])
                 for m in BANNED["mandatory"]}
        uid2label = {m["uid"]: m.get("label", m["uid"]) for m in BANNED["mandatory"]}

        for state in State:
            spec = STATES.get(state)
            if not spec or not spec.mandatory:
                continue
            variants = SCRIPTS.get("utterances", {}).get(state.value) or []
            for i, tpl in enumerate(variants):
                text = normalize(render(tpl))
                for uid in spec.mandatory:
                    self.assertIn(uid, specs, f"{state.value} 声明了未知必说话术 {uid}")
                    self.assertTrue(specs[uid].matched(text),
                                    f"{state.value} 第 {i + 1} 句不满足必说话术「{uid2label[uid]}」：{render(tpl)}")

    def test_state_fallback_satisfies_mandatory(self):
        specs = {m["uid"]: MandatorySpec(uid=m["uid"], label=m.get("label", ""), groups=m["groups"])
                 for m in BANNED["mandatory"]}
        uid2label = {m["uid"]: m.get("label", m["uid"]) for m in BANNED["mandatory"]}

        for state in State:
            spec = STATES.get(state)
            if not spec or not spec.mandatory:
                continue
            val = SCRIPTS["fallbacks"].get(spec.fallback)
            if val is None:
                continue
            items = [val] if isinstance(val, str) else list(val)
            for i, tpl in enumerate(items):
                text = normalize(render(tpl))
                for uid in spec.mandatory:
                    self.assertTrue(specs[uid].matched(text),
                                    f"{state.value} 兜底话术第 {i + 1} 句不满足必说话术「{uid2label[uid]}」：{render(tpl)}")

    def test_each_main_state_has_variants(self):
        """主要状态至少要有 2 种说法，否则整通电话像复读机。"""
        for state in (State.OPENING, State.REJECT_REASON, State.CLOSE,
                      State.OBJECTION, State.ESCALATE, State.HANDOFF, State.CLOSED):
            n = len(SCRIPTS.get("utterances", {}).get(state.value) or [])
            self.assertGreaterEqual(n, 2, f"{state.value} 只有 {n} 句，建议至少 2 句")
        # 整改动作是按场景动态变的，句数看 rectify_templates 的三个桶
        for bucket in ("needs_submit", "self_contained", "as_is"):
            n = len(SCRIPTS.get("rectify_templates", {}).get(bucket) or [])
            self.assertGreaterEqual(n, 2, f"rectify_templates.{bucket} 只有 {n} 句")

    def test_total_utterance_count(self):
        n = sum(1 for _ in iter_utterances())
        self.assertGreaterEqual(n, 45, f"话术总量 {n} 句，比预期少")

    def test_hotline_constant(self):
        """服务热线必须与业务文档一致，写错就是重大事故。"""
        self.assertEqual(SCRIPTS.get("hotline"), "4006180092")


class TestRotationVariety(unittest.IsolatedAsyncioTestCase):
    async def test_no_consecutive_repeat(self):
        runtime = load_json("config/runtime.json")
        runtime["latency_scale"] = 0.0
        eng = Engine(runtime=runtime)
        texts = [eng._utterance(State.OPENING) for _ in range(8)]
        for a, b in zip(texts, texts[1:]):
            self.assertNotEqual(a, b, "连续两次说了同一句话，轮转没生效")
        self.assertGreaterEqual(len(set(texts[:3])), 3)

    async def test_reprompts_rotate(self):
        runtime = load_json("config/runtime.json")
        runtime["latency_scale"] = 0.0
        eng = Engine(runtime=runtime)
        texts = [eng._reprompt(State.REJECT_REASON) for _ in range(6)]
        for a, b in zip(texts, texts[1:]):
            self.assertNotEqual(a, b)
        self.assertGreaterEqual(len(set(texts[:3])), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
