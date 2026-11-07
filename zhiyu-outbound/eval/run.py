"""跑评测集，输出闭环率 / 关单判定正确率 / 必说话术覆盖 / 时延分位数。

用法：
    python eval/run.py             # 按 config/runtime.json 的时延参数跑
    python eval/run.py --fast      # 时延缩放为 0，秒级（默认选项，验证逻辑）
    python eval/run.py --json out/report.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import policy
from server.engine import Engine, load_json
from server.flow import State
from server.session import build_report, format_report, TurnTimeline


def load_cases(path: str) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_ctx(raw: dict | None) -> policy.CallContext:
    raw = raw or {}
    return policy.CallContext(
        attempt_count=int(raw.get("attempt_count", 1)),
        already_called_unreached=bool(raw.get("already_called_unreached", False)),
        merchant_requested_callback=bool(raw.get("merchant_requested_callback", False)),
        escalation_risk=bool(raw.get("escalation_risk", False)),
    )


async def run_case(case: dict, runtime: dict) -> dict:
    scen = policy.get_scenario(case.get("scenario", "S001"))
    if scen is None:
        return {"id": case["id"], "ok": False, "error": f"场景不存在 {case.get('scenario')}",
                "checks": {"scenario_exists": False}, "desc": case.get("desc", ""),
                "timelines": []}

    eng = Engine(runtime=runtime, scenario=scen, call_ctx=build_ctx(case.get("ctx")))
    result = await eng.run_scripted(case.get("lines", []))
    comp = eng.compliance()
    exp = case.get("expect", {})
    decision = result.decision or {}

    checks: dict[str, bool] = {}
    if "called" in exp:
        checks["called"] = result.called == exp["called"]
    if "closed" in exp:
        checks["closed"] = result.closed == exp["closed"]
    if "method" in exp:
        checks["method"] = decision.get("method") == exp["method"]
    if "action" in exp:
        checks["action"] = decision.get("action") == exp["action"]
    if "call_result" in exp:
        checks["call_result"] = (result.writeback or {}).get("call_result") == exp["call_result"]
    if "final_state" in exp:
        checks["final_state"] = result.final_state.value == exp["final_state"]
    if exp.get("ticket") is True:
        checks["ticket"] = bool((result.writeback or {}).get("ticket_id"))
    if exp.get("no_reason") == "true":
        # 身份未确认时绝不能进入"讲驳回原因"状态
        checks["no_reason"] = all(t.state_after is not State.REJECT_REASON for t in result.turns)

    return {
        "id": case["id"],
        "desc": case.get("desc", ""),
        "scenario": scen.id,
        "called": result.called,
        "closed": result.closed,
        "final_state": result.final_state.value if result.turns else "NOT_CALLED",
        "method": decision.get("method", "-"),
        "action": decision.get("action", "-"),
        "reason_code": decision.get("reason_code", "-"),
        "turns": len(result.turns),
        "ticket": (result.writeback or {}).get("ticket_id", ""),
        "call_result": (result.writeback or {}).get("call_result", ""),
        "mandatory_missing": comp["mandatory_missing"],
        "guard_trips": comp["guard_trips"],
        "checks": checks,
        "ok": all(checks.values()) if checks else True,
        "timelines": [t.timeline for t in result.turns],
    }


async def main_async(args: argparse.Namespace) -> int:
    runtime = load_json("config/runtime.json")
    if args.fast:
        runtime["latency_scale"] = 0.0
    cases = load_cases(os.path.join(os.path.dirname(__file__), "cases.jsonl"))

    rows: list[dict] = []
    all_tl: list[TurnTimeline] = []
    for c in cases:
        r = await run_case(c, runtime)
        all_tl.extend(r.pop("timelines", []))
        rows.append(r)

    total = len(rows)
    passed = sum(1 for r in rows if r["ok"])
    called = sum(1 for r in rows if r.get("called"))
    skipped = sum(1 for r in rows if r.get("called") is False)
    closed = sum(1 for r in rows if r.get("closed"))
    ticketed = sum(1 for r in rows if r.get("ticket"))
    guard_hits = sum(r.get("guard_trips", 0) for r in rows)
    # 必说话术只在"真的外呼了"的单子上考核，判定不外呼的单子不含此项
    called_rows = [r for r in rows if r.get("called")]
    mand_ok = sum(1 for r in called_rows if not r.get("mandatory_missing"))

    print("=" * 108)
    print(f"{'case':<6}{'ok':<6}{'called':<8}{'state':<15}{'method':<10}{'action':<22}{'turns':<7}{'result':<12}{'desc'}")
    print("-" * 108)
    for r in rows:
        print(f"{r['id']:<6}{'PASS' if r['ok'] else 'FAIL':<6}{str(r.get('called','-')):<8}"
              f"{r.get('final_state','-'):<15}{r.get('method','-'):<10}{r.get('action','-'):<22}"
              f"{r.get('turns','-'):<7}{r.get('call_result') or '-':<12}{r.get('desc','')}")
        if not r["ok"]:
            print(f"       └ 未通过的检查项: {[k for k, v in r.get('checks', {}).items() if not v]}"
                  f"  缺失必说话术: {r.get('mandatory_missing')}")
    print("=" * 108)
    print(f"通过 {passed}/{total}   实际外呼 {called}   判定不外呼 {skipped}   闭环 {closed}   "
          f"已回写 {ticketed}   必说话术齐全 {mand_ok}/{len(called_rows)}   禁语拦截触发 {guard_hits}")
    print()
    print(format_report(build_report(all_tl)))
    mode = "模拟(mock)" if runtime.get("simulate", True) else "真实服务"
    print(f"\n时延口径：首字响应 = 用户说话结束 → Agent 音频开始播放。当前 provider：{mode}"
          f"，latency_scale={runtime.get('latency_scale', 1.0)}")

    report = {
        "summary": {
            "total": total, "passed": passed, "called": called, "skipped": skipped,
            "closed": closed, "ticketed": ticketed, "mandatory_ok": mand_ok,
            "mandatory_total": len(called_rows),
            "guard_trips": guard_hits,
            "latency_source": "mock_simulated" if runtime.get("simulate", True) else "measured",
            "latency_scale": runtime.get("latency_scale", 1.0),
        },
        "latency": build_report(all_tl),
        "cases": rows,
    }
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已写入 {args.json}")
    return 0 if passed == total else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="时延缩放为 0，秒级跑完")
    ap.add_argument("--json", default="eval/out/report.json", help="报告输出路径")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
