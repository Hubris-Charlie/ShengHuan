"""资质审核驳回外呼 SOP 状态机 + 槽位管理。

通话结构（来自《赛题数据说明》对业务的定义，四段式）：
    ① 自报身份 → ② 确认对方身份 → ③ 说明驳回原因 + 整改动作 → ④ 礼貌结束

核心原则
--------
1. **状态转移是纯函数**：只看槽位与意图，不看 LLM 措辞。
2. **身份确认是硬闸门**：①→③ 之间必须拿到对方确认。给错人讲驳回原因
   是重大事故，所以这里**不允许强制推进**——追问用尽就礼貌挂断。
3. 每个状态都有兜底话术，保证电话不断。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class State(str, Enum):
    OPENING = "OPENING"                  # 自报身份 + 请求确认对方身份
    VERIFY_MERCHANT = "VERIFY_MERCHANT"  # 确认对方是 XX 公司（硬闸门）
    REJECT_REASON = "REJECT_REASON"      # 说明驳回原因
    RECTIFY = "RECTIFY"                  # 整改动作 + 确认听懂
    CLOSE = "CLOSE"                      # 礼貌结束（含热线）
    CLOSED = "CLOSED"                    # 通话结束
    OBJECTION = "OBJECTION"              # 异议/质疑（旁挂，回应后回原流程）
    ESCALATE = "ESCALATE"                # 越级投诉/外部曝光风险 → 升级处理
    HANDOFF = "HANDOFF"                  # 转人工
    EXCEPTION = "EXCEPTION"              # 异常兜底


class Intent(str, Enum):
    AFFIRM = "AFFIRM"
    DENY = "DENY"
    QUESTION = "QUESTION"
    OBJECTION = "OBJECTION"
    ESCALATE = "ESCALATE"
    HANDOFF = "HANDOFF"
    UNCLEAR = "UNCLEAR"
    SILENCE = "SILENCE"


@dataclass(frozen=True)
class StateSpec:
    name: State
    goal: str
    required_slots: tuple[str, ...] = ()
    mandatory: tuple[str, ...] = ()
    on_complete: State = State.CLOSED
    fallback: str = "generic_fallback"
    barge_in: str = "stop"
    max_reprompt: int = 2
    # 追问用尽时的去处。None = 强制推进到 on_complete；
    # 身份确认这类闸门必须设成 CLOSE，绝不带着未确认的身份往下走。
    on_exhaust: State | None = None
    exhaust_reason: str = "forced_advance"


STATES: dict[State, StateSpec] = {
    State.OPENING: StateSpec(
        name=State.OPENING,
        goal="自报身份（京东商家资质审核专员）并询问对方是否为申请单上的商家公司/店铺",
        mandatory=("identity_declaration", "merchant_verification"),
        on_complete=State.VERIFY_MERCHANT,
        fallback="opening_fallback",
    ),
    State.VERIFY_MERCHANT: StateSpec(
        name=State.VERIFY_MERCHANT,
        goal="等待对方确认身份；确认后才可说明驳回原因，否认则礼貌结束",
        required_slots=("merchant_confirmed",),
        on_complete=State.REJECT_REASON,
        on_exhaust=State.CLOSE,          # 硬闸门：不确认就绝不讲驳回原因
        exhaust_reason="unverified_close",
        fallback="verify_fallback",
        max_reprompt=2,
    ),
    State.REJECT_REASON: StateSpec(
        name=State.REJECT_REASON,
        goal="准确说出不合格的具体是哪一项（对应二级分类），并说明为什么不符合",
        required_slots=("reason_ack",),
        mandatory=("reject_reason",),
        on_complete=State.RECTIFY,
        fallback="reason_fallback",
    ),
    State.RECTIFY: StateSpec(
        name=State.RECTIFY,
        goal="给出可执行的整改动作（补什么、改哪里、怎么重新提交），并确认对方听懂",
        required_slots=("rectify_ack",),
        mandatory=("rectify_action",),
        on_complete=State.CLOSE,
        fallback="rectify_fallback",
        barge_in="ack",
    ),
    State.CLOSE: StateSpec(
        name=State.CLOSE,
        goal="礼貌结束，告知服务热线，祝生活愉快",
        mandatory=("closing_courtesy", "hotline_notice"),
        on_complete=State.CLOSED,
        fallback="close_fallback",
    ),
    State.CLOSED: StateSpec(
        name=State.CLOSED,
        goal="通话结束",
        on_complete=State.CLOSED,
    ),
    State.OBJECTION: StateSpec(
        name=State.OBJECTION,
        goal="接住商家质疑（我这明明是真的授权 / 品牌方模板改不了），回应后回到原流程",
        on_complete=State.REJECT_REASON,
        fallback="objection_fallback",
    ),
    State.ESCALATE: StateSpec(
        name=State.ESCALATE,
        goal="承接越级投诉或外部曝光风险，告知将升级反馈，不做任何承诺",
        mandatory=("escalation_notice",),
        on_complete=State.CLOSED,
        fallback="escalate_fallback",
    ),
    State.HANDOFF: StateSpec(
        name=State.HANDOFF,
        goal="告知将转接人工同事并结束",
        mandatory=("handoff_notice",),
        on_complete=State.CLOSED,
        fallback="handoff_fallback",
    ),
    State.EXCEPTION: StateSpec(
        name=State.EXCEPTION,
        goal="承认系统问题并给出兜底承诺",
        on_complete=State.CLOSED,
        fallback="exception_fallback",
    ),
}


# ---------------------------------------------------------------- 槽位

@dataclass
class Slot:
    name: str
    value: Any = None
    confidence: float = 0.0
    provenance: str = ""
    pending_confirm: Any = None

    @property
    def filled(self) -> bool:
        return self.value is not None and self.value != "" and self.value is not False


class SlotStore:
    """槽位存储。改口走"确认-覆盖"两拍，避免误改。"""

    def __init__(self) -> None:
        self._slots: dict[str, Slot] = {}

    def get(self, name: str) -> Slot:
        return self._slots.setdefault(name, Slot(name))

    def value(self, name: str) -> Any:
        return self._slots[name].value if name in self._slots else None

    def set(self, name: str, value: Any, confidence: float = 1.0, provenance: str = "") -> str:
        slot = self.get(name)
        if slot.pending_confirm is not None and slot.pending_confirm == value:
            slot.value = value
            slot.confidence = confidence
            slot.provenance = provenance
            slot.pending_confirm = None
            return "set"
        if slot.value is None:
            slot.value = value
            slot.confidence = confidence
            slot.provenance = provenance
            return "set"
        if slot.value == value:
            slot.confidence = max(slot.confidence, confidence)
            return "noop"
        slot.pending_confirm = value
        return "confirm_required"

    def missing(self, names: tuple[str, ...]) -> list[str]:
        return [n for n in names if not self.get(n).filled]

    def snapshot(self) -> dict[str, Any]:
        return {
            n: {"value": s.value, "confidence": round(s.confidence, 2), "provenance": s.provenance}
            for n, s in self._slots.items()
        }

    def flat(self) -> dict[str, Any]:
        return {n: s.value for n, s in self._slots.items()}


# ---------------------------------------------------------------- 状态转移

@dataclass
class Transition:
    from_state: State
    to_state: State
    reason: str


def transition(spec: StateSpec, slots: SlotStore, intent: Intent, *, reprompt_count: int = 0) -> Transition:
    """纯函数：按槽位与意图决定下一步。顺序有意为之，不要随意调整。"""
    s = spec.name

    # 越级投诉/外部曝光优先于一切；转人工其次
    if intent is Intent.ESCALATE:
        return Transition(s, State.ESCALATE, "escalation_risk")
    if intent is Intent.HANDOFF:
        return Transition(s, State.HANDOFF, "user_requested_handoff")

    if s is State.OBJECTION:
        return Transition(s, spec.on_complete, "objection_handled")

    if s is State.OPENING:
        if intent is Intent.OBJECTION:
            return Transition(s, State.OBJECTION, "objection_at_opening")
        return Transition(s, State.VERIFY_MERCHANT, "opening_done")

    if s is State.VERIFY_MERCHANT:
        # 打错人：立即礼貌结束，绝不继续讲驳回原因
        if intent is Intent.DENY:
            return Transition(s, State.CLOSE, "wrong_merchant")
        if intent is Intent.OBJECTION:
            return Transition(s, State.OBJECTION, "objection_at_verify")

    if intent is Intent.OBJECTION and s in (State.REJECT_REASON, State.RECTIFY):
        return Transition(s, State.OBJECTION, "objection_raised")

    missing = slots.missing(spec.required_slots)
    if missing:
        if reprompt_count >= spec.max_reprompt:
            if spec.on_exhaust is not None:
                return Transition(s, spec.on_exhaust, spec.exhaust_reason)
            return Transition(s, spec.on_complete, f"forced_advance_missing={','.join(missing)}")
        return Transition(s, s, f"await_slots={','.join(missing)}")

    return Transition(s, spec.on_complete, "slots_complete")
