"""会话事件流与时延埋点。

设计要点
--------
1. 每个事件都是一条 JSONL，带 monotonic 相对时间 t（秒，相对通话开始）。
2. 音频层的"生成进度"与"播放进度"分开记，才能算准打断时的已播长度。
3. 报告只依赖 TurnTimeline 的字段，不依赖具体 provider。
"""
from __future__ import annotations

import json
import os
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


def now() -> float:
    return time.perf_counter()


def wall() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class EventLog:
    """追加式事件流：内存 + 可选落盘 JSONL。"""

    def __init__(self, session_id: str | None = None, path: str | None = None):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.path = path
        self.events: list[dict[str, Any]] = []
        self._t0 = now()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            open(path, "w", encoding="utf-8").close()  # 每通电话新文件

    def t(self) -> float:
        return round(now() - self._t0, 4)

    def emit(self, layer: str, type_: str, **payload: Any) -> dict[str, Any]:
        ev = {
            "t": self.t(),
            "wall": wall(),
            "session": self.session_id,
            "layer": layer,
            "type": type_,
            **payload,
        }
        self.events.append(ev)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        return ev

    def dump(self) -> list[dict[str, Any]]:
        return list(self.events)


@dataclass
class TurnTimeline:
    """一轮对话的时间线。所有字段都是相对通话开始的秒数。"""

    turn: int
    t_speech_start: float | None = None
    t_speech_end: float | None = None
    t_asr_final: float | None = None
    t_llm_req: float | None = None
    t_llm_first_token: float | None = None
    t_tts_first_byte: float | None = None
    t_playout_start: float | None = None
    t_playout_end: float | None = None
    t_barge_in: float | None = None

    # --- 分跳耗时（毫秒） ---
    def endpointing_ms(self) -> float | None:
        if self.t_speech_end is None or self.t_speech_start is None:
            return None
        return (self.t_speech_end - self.t_speech_start) * 1000

    def asr_ms(self) -> float | None:
        if self.t_asr_final is None or self.t_speech_end is None:
            return None
        return (self.t_asr_final - self.t_speech_end) * 1000

    def llm_ttft_ms(self) -> float | None:
        if self.t_llm_first_token is None or self.t_llm_req is None:
            return None
        return (self.t_llm_first_token - self.t_llm_req) * 1000

    def tts_ttfb_ms(self) -> float | None:
        if self.t_tts_first_byte is None or self.t_llm_first_token is None:
            return None
        return (self.t_tts_first_byte - self.t_llm_first_token) * 1000

    def first_response_ms(self) -> float | None:
        """**主指标**：用户说完 → Agent 声音真正开始播。

        注意：用 t_playout_start 而不是 t_tts_first_byte。
        WebRTC 抖动缓冲和播放缓冲都算在这里面，不能漏。
        """
        if self.t_playout_start is None or self.t_speech_end is None:
            return None
        return (self.t_playout_start - self.t_speech_end) * 1000

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            first_response_ms=_r(self.first_response_ms()),
            endpointing_ms=_r(self.endpointing_ms()),
            asr_ms=_r(self.asr_ms()),
            llm_ttft_ms=_r(self.llm_ttft_ms()),
            tts_ttfb_ms=_r(self.tts_ttfb_ms()),
        )
        return d


def _r(v: float | None, nd: int = 1) -> float | None:
    return None if v is None else round(v, nd)


def percentile(values: list[float], p: float) -> float | None:
    """线性插值分位数（p 取 0~100）。"""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


@dataclass
class LatencyReport:
    name: str
    n: int = 0
    p50: float | None = None
    p95: float | None = None
    mean: float | None = None
    min: float | None = None
    max: float | None = None
    unit: str = "ms"

    @classmethod
    def build(cls, name: str, values: list[float], unit: str = "ms") -> "LatencyReport":
        vals = [v for v in values if v is not None]
        if not vals:
            return cls(name=name, n=0, unit=unit)
        return cls(
            name=name,
            n=len(vals),
            p50=_r(percentile(vals, 50)),
            p95=_r(percentile(vals, 95)),
            mean=_r(statistics.fmean(vals)),
            min=_r(min(vals)),
            max=_r(max(vals)),
            unit=unit,
        )


def build_report(timelines: list[TurnTimeline]) -> dict[str, Any]:
    return {
        "turns": len(timelines),
        "first_response": asdict(LatencyReport.build("first_response", [t.first_response_ms() for t in timelines if t.first_response_ms() is not None])),
        "endpointing": asdict(LatencyReport.build("endpointing", [t.endpointing_ms() for t in timelines if t.endpointing_ms() is not None])),
        "asr": asdict(LatencyReport.build("asr", [t.asr_ms() for t in timelines if t.asr_ms() is not None])),
        "llm_ttft": asdict(LatencyReport.build("llm_ttft", [t.llm_ttft_ms() for t in timelines if t.llm_ttft_ms() is not None])),
        "tts_ttfb": asdict(LatencyReport.build("tts_ttfb", [t.tts_ttfb_ms() for t in timelines if t.tts_ttfb_ms() is not None])),
    }


def format_report(rep: dict[str, Any]) -> str:
    rows = [
        ("首字响应 (speech_end→playout)", rep["first_response"]),
        ("端点检测", rep["endpointing"]),
        ("ASR 收尾", rep["asr"]),
        ("LLM 首 token", rep["llm_ttft"]),
        ("TTS 首包", rep["tts_ttfb"]),
    ]
    lines = [f"turns={rep['turns']}", f"{'指标':<34}{'n':>4}{'P50':>10}{'P95':>10}{'mean':>10}"]
    for name, r in rows:
        if not r or r["n"] == 0:
            lines.append(f"{name:<34}{'-':>4}{'-':>10}{'-':>10}{'-':>10}")
            continue
        lines.append(
            f"{name:<34}{r['n']:>4}{r['p50']:>10.1f}{r['p95']:>10.1f}{r['mean']:>10.1f}"
        )
    return "\n".join(lines)
