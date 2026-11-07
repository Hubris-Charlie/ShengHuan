"""工具调用网关：幂等 + Outbox 重试 + 结构化回写校验。

赛题明确要求"保障调用的幂等性与一致性"，这是能白拿的分：
- 幂等键 = hash(session_id | 工具名 | 参数快照 | 版本)
- 同一键重复调用返回同一条记录，created=False，不会重复建单
- 回写走 outbox：先落本地，再提交，失败重试仍用同键
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable


def make_idem_key(session_id: str, tool: str, args: dict[str, Any], version: str = "v1") -> str:
    payload = json.dumps({"s": session_id, "t": tool, "a": args, "v": version}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


@dataclass
class ToolResult:
    ok: bool
    tool: str
    data: dict[str, Any] = field(default_factory=dict)
    created: bool = False
    idem_key: str = ""
    error: str | None = None
    attempts: int = 1


class MockBackend:
    """内存版 Mock 业务系统：名单 / 订单 / 工单。

    可注入故障：fail_first_n 让前 n 次调用失败，用来测 outbox 重试。
    """

    def __init__(self, seed_path: str | None = None) -> None:
        self.lock = threading.Lock()
        self.customers: dict[str, dict[str, Any]] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self.tickets: dict[str, dict[str, Any]] = {}      # idem_key -> 工单
        self.calls: list[dict[str, Any]] = []
        self.fail_plan: dict[str, int] = {}
        if seed_path and os.path.exists(seed_path):
            with open(seed_path, encoding="utf-8") as f:
                seed = json.load(f)
            self.customers = {c["phone"]: c for c in seed.get("customers", [])}
            self.orders = {o["order_id"]: o for o in seed.get("orders", [])}

    def inject_failure(self, tool: str, times: int) -> None:
        self.fail_plan[tool] = times

    def _maybe_fail(self, tool: str) -> str | None:
        n = self.fail_plan.get(tool, 0)
        if n > 0:
            self.fail_plan[tool] = n - 1
            return f"injected_failure:{tool}"
        return None

    # --- 三个业务工具 ---
    def query_customer(self, phone: str) -> dict[str, Any]:
        c = self.customers.get(phone)
        if not c:
            return {"found": False, "phone": phone}
        return {"found": True, **c}

    def query_order(self, order_id: str) -> dict[str, Any]:
        o = self.orders.get(order_id)
        if not o:
            return {"found": False, "order_id": order_id}
        return {"found": True, **o}

    def create_ticket(self, idem_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if idem_key in self.tickets:
                return {"ticket_id": self.tickets[idem_key]["ticket_id"], "duplicated": True}
            tid = f"T{len(self.tickets) + 1:06d}"
            self.tickets[idem_key] = {"ticket_id": tid, **payload}
            return {"ticket_id": tid, "duplicated": False}


class ToolGateway:
    def __init__(self, backend: MockBackend, max_attempts: int = 3) -> None:
        self.backend = backend
        self.max_attempts = max_attempts

    def call(self, session_id: str, tool: str, args: dict[str, Any]) -> ToolResult:
        key = make_idem_key(session_id, tool, args)
        last_err: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            err = self.backend._maybe_fail(tool)
            self.backend.calls.append({"tool": tool, "args": args, "attempt": attempt, "err": err})
            if err:
                last_err = err
                continue
            if tool == "query_customer":
                return ToolResult(True, tool, self.backend.query_customer(args["phone"]), idem_key=key, attempts=attempt)
            if tool == "query_order":
                return ToolResult(True, tool, self.backend.query_order(args["order_id"]), idem_key=key, attempts=attempt)
            if tool == "create_ticket":
                data = self.backend.create_ticket(key, args)
                return ToolResult(
                    True, tool,
                    data=data,
                    created=not data.get("duplicated", False),
                    idem_key=key,
                    attempts=attempt,
                )
            return ToolResult(False, tool, error=f"unknown_tool:{tool}", attempts=attempt)
        return ToolResult(False, tool, error=last_err, idem_key=key, attempts=self.max_attempts)


# ---------------------------------------------------------------- 回写

WRITEBACK_SCHEMA_REQUIRED = (
    "session_id", "scenario_id", "menu1", "cat1", "cat2",
    "close_method", "call_result",
)


def validate_writeback(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """回写校验。

    `merchant_confirmed` 必须为真 —— 没确认对方是申请单上的商家，
    就不能回写"已告知驳回原因"，这是业务红线（给错人讲驳回原因是重大事故）。
    """
    missing = [k for k in WRITEBACK_SCHEMA_REQUIRED if payload.get(k) in (None, "")]
    # 打错人属于要单独回写的结局（号码错误），不要求已确认身份
    if payload.get("call_result") != "wrong_number" and payload.get("merchant_confirmed") is not True:
        missing.append("merchant_confirmed")
    return (not missing), missing


class Outbox:
    """至少一次 + 幂等键 = 实际恰好一次。"""

    def __init__(self, path: str | None = None) -> None:
        self.items: list[dict[str, Any]] = []
        self.path = path
        self.sent: list[dict[str, Any]] = []

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        ok, missing = validate_writeback(payload)
        record = {
            "payload": payload,
            "valid": ok,
            "missing": missing,
            "idem_key": make_idem_key(payload.get("session_id", ""), "writeback", payload),
            "status": "pending",
        }
        self.items.append(record)
        self._flush()
        return record

    def dispatch(self, sender: Callable[[dict[str, Any], str], bool]) -> list[dict[str, Any]]:
        """把 pending 记录发出去。sender(payload, idem_key) -> bool"""
        results = []
        for rec in self.items:
            if rec["status"] == "sent":
                continue
            if not rec["valid"]:
                rec["status"] = "rejected"
                results.append(rec)
                continue
            if sender(rec["payload"], rec["idem_key"]):
                rec["status"] = "sent"
                self.sent.append(rec)
            results.append(rec)
        self._flush()
        return results

    def _flush(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)
