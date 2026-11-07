"""关单方式判定：这次到底该不该打电话。

依据《资审关单场景话术》主表里的「关单总原则」四条：

1. 关单方式为电话的，需外呼告知驳回原因与处理方案；京麦驳回场景不做强制回电限制
2. 关单方式为电话的，首次外呼未联系上，京麦驳回告知原因即可；
   相同场景二次提交错误且电话未联系上，需创事件跟进预约二次外呼；
   三次及以上相同场景信息错误，一律京麦驳回
3. 商家主动要求回电告知原因的，均需电话外呼
4. 已存在威胁越级投诉、外部曝光（微博/小红书/其他媒体平台等）的，均需回电

判定顺序很重要：规则 3、4 是**压倒性**的，即使关单方式是京麦驳回也必须回电。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOTLINE = "4006180092"


class CloseMethod(str, Enum):
    PHONE = "电话"
    JINGMAI = "京麦驳回"


class NextAction(str, Enum):
    OUTBOUND = "外呼"
    SCHEDULE_SECOND_CALL = "创事件+预约二次外呼"
    JINGMAI_REJECT = "京麦驳回"
    OUTBOUND_ESCALATED = "外呼+升级处理"


@dataclass(frozen=True)
class Scenario:
    """一个驳回场景（对应主表一行）。"""

    id: str
    menu1: str          # 一级菜单：品牌 / 入驻 / 行业资质 / 公司资质 / 商品资质
    cat1: str           # 一级分类
    cat2: str           # 二级分类
    close_method: CloseMethod
    script: str         # 参考话术模板（含【】占位符）

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.menu1, self.cat1, self.cat2)

    @property
    def label(self) -> str:
        return f"{self.menu1}/{self.cat1}/{self.cat2}"


@dataclass(frozen=True)
class CallContext:
    """一次判定所需的业务上下文。"""

    attempt_count: int = 1              # 相同场景错误提交到第几次（1/2/3+）
    already_called_unreached: bool = False   # 是否已外呼过且未接通
    merchant_requested_callback: bool = False
    escalation_risk: bool = False       # 威胁越级投诉 / 外部曝光


@dataclass(frozen=True)
class CallDecision:
    should_call: bool
    method: CloseMethod
    action: NextAction
    reason_code: str
    reason_text: str
    need_escalation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_call": self.should_call,
            "method": self.method.value,
            "action": self.action.value,
            "reason_code": self.reason_code,
            "reason_text": self.reason_text,
            "need_escalation": self.need_escalation,
        }


def decide_call(scenario: Scenario, ctx: CallContext | None = None) -> CallDecision:
    """纯函数：按关单总原则判定。顺序不可随意调换。"""
    ctx = ctx or CallContext()

    # 规则 4：越级投诉 / 外部曝光 —— 压倒一切，必须回电并升级
    if ctx.escalation_risk:
        return CallDecision(
            True, CloseMethod.PHONE, NextAction.OUTBOUND_ESCALATED,
            "escalation_risk", "存在越级投诉或外部曝光风险，一律回电并升级处理", True,
        )

    # 规则 3：商家主动要求回电
    if ctx.merchant_requested_callback:
        return CallDecision(
            True, CloseMethod.PHONE, NextAction.OUTBOUND,
            "merchant_requested", "商家主动要求回电告知原因",
        )

    # 规则 2（后半）：三次及以上，一律京麦驳回
    if ctx.attempt_count >= 3:
        return CallDecision(
            False, CloseMethod.JINGMAI, NextAction.JINGMAI_REJECT,
            "third_or_more", "相同场景三次及以上提交错误，一律京麦驳回",
        )

    # 京麦驳回场景：不做强制回电
    if scenario.close_method is CloseMethod.JINGMAI:
        return CallDecision(
            False, CloseMethod.JINGMAI, NextAction.JINGMAI_REJECT,
            "jingmai_scene", "该场景关单方式为京麦驳回，不做强制回电",
        )

    # 规则 2（前半）：电话场景，看是第几次、上次是否接通
    if ctx.already_called_unreached:
        if ctx.attempt_count >= 2:
            return CallDecision(
                True, CloseMethod.PHONE, NextAction.SCHEDULE_SECOND_CALL,
                "second_attempt_schedule",
                "相同场景二次提交仍错且电话未联系上，需创事件跟进预约二次外呼",
            )
        return CallDecision(
            False, CloseMethod.JINGMAI, NextAction.JINGMAI_REJECT,
            "first_unreached", "电话场景首次外呼未联系上，京麦驳回告知原因即可",
        )

    # 电话场景，正常外呼
    return CallDecision(
        True, CloseMethod.PHONE, NextAction.OUTBOUND,
        "phone_scene", "该场景关单方式为电话，需外呼告知驳回原因与处理方案",
    )


# ---------------------------------------------------------------- 场景库

_CACHE: dict[str, Scenario] | None = None


def load_scenarios(path: str | None = None) -> dict[str, Scenario]:
    """加载 config/scenarios.json。已处理主表的纵向合并，三级分类是填满的。"""
    global _CACHE
    if _CACHE is not None and path is None:
        return _CACHE
    p = path or os.path.join(ROOT, "config", "scenarios.json")
    if not os.path.exists(p):
        _CACHE = {}
        return _CACHE
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, Scenario] = {}
    for r in raw:
        method_raw = (r.get("close_method") or "").strip()
        method = CloseMethod.PHONE if method_raw == "电话" else CloseMethod.JINGMAI
        out[r["id"]] = Scenario(
            id=r["id"],
            menu1=r.get("menu1", ""),
            cat1=r.get("cat1", ""),
            cat2=r.get("cat2", ""),
            close_method=method,
            script=r.get("script", ""),
        )
    if path is None:
        _CACHE = out
    return out


def find_scenario(menu1: str, cat1: str, cat2: str) -> Scenario | None:
    for s in load_scenarios().values():
        if s.menu1 == menu1 and s.cat1 == cat1 and s.cat2 == cat2:
            return s
    return None


def phone_scenarios() -> list[Scenario]:
    return [s for s in load_scenarios().values() if s.close_method is CloseMethod.PHONE]


def get_scenario(scenario_id: str) -> Scenario | None:
    return load_scenarios().get(scenario_id)


# ---------------------------------------------------------------- 原因 / 整改 口语化

_RA: dict[str, Any] | None = None
_PAREN_RE = re.compile(r"[（(][^）)]*[）)]")


def load_reason_action(path: str | None = None) -> dict[str, Any]:
    """加载 config/reason_action.json：按一级分类给口播化的整改动作。"""
    global _RA
    if _RA is not None and path is None:
        return _RA
    p = path or os.path.join(ROOT, "config", "reason_action.json")
    data: dict[str, Any] = {"actions": {}, "reason_overrides": {}}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        data.pop("_note", None)
    if path is None:
        _RA = data
    return data


def clean_reason(cat2: str) -> str:
    """把二级分类变成能直接念出来的驳回原因。

    原始分类里带括号补充说明（"授权书内容不完整（包含盖章…）"），
    念出来会很啰嗦，这里去掉括号部分。全空的兜底成"暂不招商或受管控"。
    """
    s = (cat2 or "").replace("\n", "")
    s = _PAREN_RE.sub("", s).strip("，,、（(")
    vague = {
        "其他": "该品牌暂不招商，或受平台管控",
        "其它": "该品牌暂不招商，或受平台管控",
        "不符合业务要求": "该品牌暂不招商，或受平台管控",
        "资质造假": "提供的资质材料存在失真，与实际不符",
    }
    if not s:
        return "该品牌暂不招商，或受平台管控"
    return vague.get(s, s)


def reason_text(scenario: Scenario | None) -> str:
    if scenario is None:
        return "提交的资质材料不符合申请标准"
    return clean_reason(scenario.cat2)


_ACTION_START = re.compile(
    r"(?:需您|需要您|辛苦您|建议您|麻烦您|请您|故请您"
    r"|请(?=补齐|提供|修改|删除|核实|确认|重新|规范|上传|选择|开具|补充|调整|更换|更改|【)"
    r"|需(?=上传|提供|修改|补齐|重新|确认|选择|及时|开具)"
    r"|建议(?=更换|重新))"
)
_HEAD_RE = re.compile(r"^.*?(?:京东资质审核专员|京东商家资质审核专员)[，,]?")
_TAIL_RE = re.compile(r"[，,]?如有疑问.*$")
_POLITE_RE = re.compile(
    r"^(?:请|故请您|需您|需要您|辛苦您|麻烦您|建议您"
    r"|需(?=上传|提供|修改|补齐|重新|确认|选择|及时|开具))"
)
_BOILER_RE = re.compile(r"[。！]?\s*(?:故本单需为您驳回修改|给予驳回|故驳回|故平台驳回申请)[。！]?$")


def split_action(script: str) -> str:
    """从官方参考话术里拆出整改动作。拆不出或带空占位符就返回空串。

    官方话术是"原因 + 整改 + 热线"揉在一句里的，这里只取整改那一段，
    供 RECTIFY 状态单独播报。
    """
    if not script:
        return ""
    s = _TAIL_RE.sub("", script.strip())
    s = _HEAD_RE.sub("", s).strip("，,。 ")
    s = _BOILER_RE.sub("", s).strip("，,。 ")
    m = _ACTION_START.search(s)
    if not m:
        return ""
    core = _POLITE_RE.sub("", s[m.start():]).strip("，,。 ")
    core = _PAREN_RE.sub("", core).strip("，,。 ")
    # 带占位符/人名占位的没法直接念，交给分类兜底
    if not core or "【" in core or "XX" in core or "XXX" in core:
        return ""
    return core


def action_options(scenario: Scenario | None) -> list[str]:
    """整改动作候选：官方话术能干净拆出的优先，否则用一级分类的兜底说法。"""
    if scenario is None:
        return []
    official = split_action(scenario.script)
    if official:
        return [official]
    ra = load_reason_action()
    return list((ra.get("actions") or {}).get(f"{scenario.menu1}/{scenario.cat1}") or [])

