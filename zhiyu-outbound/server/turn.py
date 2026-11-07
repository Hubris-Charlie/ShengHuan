"""打断控制与播放进度记账。

这是最容易埋雷的一块，两个必须分清的量：
- **生成进度**：TTS 已经合成出多少文本/音频（往往领先一大截）
- **播放进度**：扬声器真正播出去多少毫秒

打断时"我已经说到哪儿了"必须按**播放进度**算。按生成进度算的话，
你会以为已经说完了整句，实际上用户只听到一半——下一轮 Agent 就会
重复或答错。这就是很多 demo 里"打断之后答非所问"的根因。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AudioChunk:
    seq: int
    text: str
    dur_ms: float


@dataclass
class BargeResult:
    """一次打断的结果。"""

    triggered: bool
    reason: str
    played_ms: float
    spoken_prefix: str          # 用户真正听到的那部分
    unspoken: str               # 生成了但没播出去的部分（不应计入对话历史）
    generated_total_ms: float


class TurnController:
    """管理一轮 Agent 发声。生成与播放两条进度线。"""

    def __init__(self) -> None:
        self.chunks: list[AudioChunk] = []
        self.played_ms: float = 0.0
        self.gen_done: bool = False
        self.cancelled: bool = False

    # ---- 生成侧 ----
    def on_tts_chunk(self, text: str, dur_ms: float) -> AudioChunk:
        if self.cancelled:
            return AudioChunk(seq=-1, text="", dur_ms=0.0)
        c = AudioChunk(seq=len(self.chunks), text=text, dur_ms=float(dur_ms))
        self.chunks.append(c)
        return c

    def on_gen_done(self) -> None:
        self.gen_done = True

    # ---- 播放侧 ----
    def on_playout(self, played_ms: float) -> None:
        self.played_ms = max(self.played_ms, float(played_ms))

    # ---- 计算 ----
    @property
    def generated_total_ms(self) -> float:
        return sum(c.dur_ms for c in self.chunks)

    def spoken_prefix(self) -> str:
        """按播放进度裁剪出"已播出"的文本。"""
        left = self.played_ms
        out: list[str] = []
        for c in self.chunks:
            if left <= 0:
                break
            if left >= c.dur_ms:
                out.append(c.text)
                left -= c.dur_ms
            else:
                ratio = left / c.dur_ms if c.dur_ms > 0 else 1.0
                n = int(len(c.text) * ratio)
                out.append(c.text[:n])
                left = 0
        return "".join(out)

    def unspoken(self) -> str:
        spoken = self.spoken_prefix()
        full = "".join(c.text for c in self.chunks)
        return full[len(spoken):] if full.startswith(spoken) else ""

    # ---- 打断 ----
    def barge_in(self) -> BargeResult:
        self.cancelled = True
        return BargeResult(
            triggered=True,
            reason="user_speech_start",
            played_ms=self.played_ms,
            spoken_prefix=self.spoken_prefix(),
            unspoken=self.unspoken(),
            generated_total_ms=self.generated_total_ms,
        )

    def reset(self) -> None:
        self.chunks = []
        self.played_ms = 0.0
        self.gen_done = False
        self.cancelled = False


# ---------------------------------------------------------------- 回声闸门

class EchoGate:
    """防止机器人被自己的 TTS 回声触发打断。

    判据（三条全满足才认为是一次真实打断）：
    1. 输入语音持续时长 ≥ min_speech_ms（滤掉爆音、喷麦）
    2. 与当前播放信号的相似度 < echo_ratio（判为回声则丢弃）
    3. 距离 TTS 起播已超过 suppress_after_onset_ms（滤掉起播瞬态）
    """

    def __init__(
        self,
        min_speech_ms: float = 180.0,
        echo_ratio: float = 0.80,
        suppress_after_onset_ms: float = 120.0,
    ) -> None:
        self.min_speech_ms = min_speech_ms
        self.echo_ratio = echo_ratio
        self.suppress_after_onset_ms = suppress_after_onset_ms
        self.agent_speaking = False
        self.speech_onset_ms: float | None = None
        self.dropped = 0

    def set_agent_speaking(self, speaking: bool, onset_ms: float | None = None) -> None:
        self.agent_speaking = speaking
        self.speech_onset_ms = onset_ms

    def should_trigger(
        self,
        *,
        now_ms: float,
        speech_ms: float,
        echo_score: float = 0.0,
    ) -> tuple[bool, str]:
        if not self.agent_speaking:
            # Agent 没在说话，这就是普通一轮输入，不是打断
            return False, "not_speaking"
        if speech_ms < self.min_speech_ms:
            self.dropped += 1
            return False, f"too_short({speech_ms:.0f}ms)"
        if self.speech_onset_ms is not None and (now_ms - self.speech_onset_ms) < self.suppress_after_onset_ms:
            self.dropped += 1
            return False, "onset_suppressed"
        if echo_score >= self.echo_ratio:
            self.dropped += 1
            return False, f"self_echo({echo_score:.2f})"
        return True, "ok"
