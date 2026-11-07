"""可插拔的 ASR / TTS / LLM 提供方。

设计目的：在没有云密钥、没有 GPU 的机器上，也能把**整条管线**跑通并产出真实埋点；
同时把真实服务的接入点固定下来，接密钥就能切。

- `Mock*`：本地合成，零网络依赖，可注入模拟时延
- `Cloud*`：真实接入的最小实现（OpenAI 兼容 LLM 已可用；ASR/TTS 为适配骨架）
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import struct
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Protocol

SAMPLE_RATE = 16000


# ---------------------------------------------------------------- 接口

class LLMClient(Protocol):
    async def stream(self, messages: list[dict[str, str]], **kw: Any) -> AsyncIterator[str]: ...


class ASRClient(Protocol):
    async def transcribe(self, pcm: bytes) -> str: ...


class TTSClient(Protocol):
    async def stream(self, text: str) -> AsyncIterator[bytes]: ...


# ---------------------------------------------------------------- 音频工具

def estimate_duration_ms(text: str, char_ms: float = 165.0) -> float:
    """按字数估算播报时长（中文经验值 ≈165ms/字）。"""
    return max(120.0, len(text) * char_ms)


def synth_tone(text: str, char_ms: float = 165.0, sr: int = SAMPLE_RATE, freq: float = 220.0) -> bytes:
    """把文本合成为一段可播放的 PCM16 单声道音频。

    这不是真的语音，只是一个**占位音**：让浏览器通路、播放进度、打断逻辑
    在没有云密钥时也能被真实地端到端验证。接真实 TTS 后直接替换。
    """
    total_ms = estimate_duration_ms(text, char_ms)
    n = int(sr * total_ms / 1000.0)
    buf = bytearray()
    amp = 0.18 * 32767
    for i in range(n):
        t = i / sr
        # 轻微起伏，避免完全单调，便于人耳判断"还在播"
        env = 0.55 + 0.45 * math.sin(2 * math.pi * 0.8 * t)
        v = int(amp * env * math.sin(2 * math.pi * freq * t))
        buf += struct.pack("<h", max(-32768, min(32767, v)))
    return bytes(buf)


def silence(ms: float, sr: int = SAMPLE_RATE) -> bytes:
    return b"\x00\x00" * int(sr * ms / 1000.0)


def pcm16_rms(pcm: bytes) -> float:
    """计算 PCM16 的 RMS 能量（0~1）。用于最简 VAD。"""
    if not pcm:
        return 0.0
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    vals = struct.unpack("<" + "h" * n, pcm[: n * 2])
    s = sum(v * v for v in vals) / n
    return math.sqrt(s) / 32768.0


def pcm16_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE, channels: int = 1) -> bytes:
    """把裸 PCM16 包成 WAV。几乎所有 ASR 接口都吃这个格式。"""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class HttpASR:
    """OpenAI 兼容的 `/audio/transcriptions` 接口。

    这一类接口覆盖面很广（SiliconFlow、DashScope 兼容模式、Groq、OpenAI，
    以及各种自建 whisper 服务），都是"给一段音频、返回一段文本"，
    正好匹配我们"VAD 切句 → 整句识别"的流程，实现成本最低。

    热词偏置：Whisper 系接口支持 `prompt` 参数，把领域词表塞进去即可。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "",
        language: str = "zh",
        hotwords: Iterable[str] = (),
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("ASR_API_KEY", "")
        self.model = model
        self.language = language
        self.hotwords = [h for h in hotwords if h]
        self.timeout = timeout

    def _prompt(self) -> str:
        if not self.hotwords:
            return ""
        # 控制长度，太长反而降低效果（这是热词偏置的通病）
        return "以下为专有名词，请优先按此识别：" + "、".join(self.hotwords[:40])

    async def transcribe(self, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> str:
        import httpx

        if not pcm:
            return ""
        wav = pcm16_to_wav(pcm, sample_rate)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data: dict[str, str] = {"model": self.model, "language": self.language, "response_format": "json"}
        prompt = self._prompt()
        if prompt:
            data["prompt"] = prompt
        files = {"file": ("utterance.wav", wav, "audio/wav")}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/audio/transcriptions", headers=headers, files=files, data=data)
            r.raise_for_status()
            obj = r.json()
        text = obj.get("text") if isinstance(obj, dict) else None
        return (text or "").strip()


# ---------------------------------------------------------------- Mock 实现

@dataclass
class MockLatency:
    """模拟各环节耗时，用于在没有真实服务时验证埋点链路。

    scale：时延缩放系数。1.0 = 按配置的真实量级等待；<=0 = 完全不等待（单测用）。
    注意：这里必须允许"完全不等待"，否则单测会被 mock 的 sleep 拖成分钟级。
    """

    llm_ttft_ms: tuple[float, float] = (260.0, 520.0)
    tokens_per_sec: float = 55.0
    tts_ttfb_ms: tuple[float, float] = (140.0, 260.0)
    tts_char_ms: float = 165.0
    scale: float = 1.0

    def _mid(self, rng: tuple[float, float], seed: float) -> float:
        lo, hi = rng
        return lo + (hi - lo) * (0.5 + 0.5 * math.sin(seed))

    def delay_s(self, ms: float) -> float:
        if self.scale <= 0:
            return 0.0
        return (ms / 1000.0) / self.scale


class MockLLM:
    """按"给定回复"逐 token 吐出，模拟真流式。"""

    def __init__(self, latency: MockLatency | None = None, reply: str = "") -> None:
        self.latency = latency or MockLatency()
        self.reply = reply

    async def stream(self, messages: list[dict[str, str]], **kw: Any) -> AsyncIterator[str]:
        # 真实实现里这里会调模型；Mock 直接用调用方指定的回复
        reply = kw.get("mock_reply", self.reply) or ""
        ttft = self.latency._mid(self.latency.llm_ttft_ms, len(reply))
        d = self.latency.delay_s(ttft)
        if d:
            await asyncio.sleep(d)
        if not reply:
            return
        per_tok_ms = 1000.0 / max(1.0, self.latency.tokens_per_sec)
        step = self.latency.delay_s(per_tok_ms)
        for ch in reply:
            yield ch
            if step:
                await asyncio.sleep(step)


class MockASR:
    """把调用方给定的文本当作识别结果。"""

    def __init__(self, latency_ms: tuple[float, float] = (80, 180)) -> None:
        self.latency_ms = latency_ms

    async def transcribe(self, pcm: bytes) -> str:
        await asyncio.sleep(sum(self.latency_ms) / 2 / 1000.0)
        return "[模拟识别]"


class MockTTS:
    def __init__(self, ttfb_ms: tuple[float, float] = (140.0, 260.0), char_ms: float = 165.0, scale: float = 1.0) -> None:
        self.ttfb_ms = ttfb_ms
        self.char_ms = char_ms
        self.scale = scale

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        if self.scale > 0:
            await asyncio.sleep((sum(self.ttfb_ms) / 2 / 1000.0) / self.scale)
        # 按子句切分流式返回，模拟真实首包提前
        for piece in _split_clause(text):
            yield synth_tone(piece, self.char_ms)
            if self.scale > 0:
                await asyncio.sleep(0.01)


def _split_clause(text: str, max_len: int = 14) -> list[str]:
    """子句级切分：首包尽快出，不等整段合成完。"""
    out: list[str] = []
    cur = ""
    for ch in text:
        cur += ch
        if ch in "，。？！、；：,.?!" or len(cur) >= max_len:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out or [text]


# ---------------------------------------------------------------- 真实服务

class OpenAICompatLLM:
    """任何 OpenAI 兼容的 /v1/chat/completions 流式接口。"""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.timeout = timeout

    async def stream(self, messages: list[dict[str, str]], **kw: Any) -> AsyncIterator[str]:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": kw.get("temperature", 0.3),
            "max_tokens": kw.get("max_tokens", 200),
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", headers=headers, json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0].get("delta", {}).get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta


class CloudASRStub:
    """真实 ASR 接入骨架（以 DashScope 实时识别为例）。

    接法（M1 要落地的部分）：
    1. 用 websockets 连到厂商的实时识别端点，握手时带上 model / 热词表
    2. 每 20~40ms 送一帧 PCM16@16k
    3. 收 partial / final 事件，final 落在本类的缓冲区里
    4. transcribe() 返回最后一次 final
    """

    def __init__(self, vendor: str, appkey: str, hotwords: Iterable[str] = (), model: str = "") -> None:
        self.vendor = vendor
        self.appkey = appkey
        self.hotwords = list(hotwords)
        self.model = model

    async def transcribe(self, pcm: bytes) -> str:
        raise NotImplementedError(
            "CloudASRStub 需要按厂商协议实现。M1 目标：先跑通 Paraformer 实时流式 + 热词。"
        )


class CloudTTSStub:
    """真实 TTS 接入骨架（以 CosyVoice 流式合成为例）。"""

    def __init__(self, vendor: str, api_key: str, voice: str = "", model: str = "") -> None:
        self.vendor = vendor
        self.api_key = api_key
        self.voice = voice
        self.model = model

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        raise NotImplementedError(
            "CloudTTSStub 需要按厂商协议实现。M1 目标：先跑通 CosyVoice 流式首包。"
        )
        yield b""  # pragma: no cover
