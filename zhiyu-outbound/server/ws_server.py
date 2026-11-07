"""浏览器实时音频通路（最小可用版，只依赖 `websockets`）。

为什么不用 Pipecat：Python 3.14 上 Pipecat 的 C 扩展依赖（PyAV 等）暂时没有
现成 wheel。这里先用「浏览器 PCM ⇄ WebSocket」把**通话链路、打断、埋点**三件事
跑实，接口保持与 Pipecat 的思路一致（AudioSource / AudioSink / 流式 TTS），
等环境允许再平移过去。

协议
----
浏览器 → 服务端
  binary                     PCM16 单声道 16kHz 音频帧
  {"type":"audio_start"}     用户开始说话
  {"type":"audio_end"}       用户说完了 —— 服务端做识别并回应
  {"type":"barge_in", "played_ms": 1234}   用户打断，报告真实已播毫秒数
  {"type":"user_text", "text":"…"}         键盘输入（等价于识别结果，便于无 ASR 时验证）
  {"type":"hangup"}

服务端 → 浏览器
  {"type":"ready", "session": "…"}
  {"type":"state", "state": "IDENTITY", "turn": 1}
  {"type":"user", "text": "…"}
  {"type":"agent", "text": "…", "fallback": false, "guard": null, "audio_ms": 2100}
  binary                     即将播放的 PCM16@16k
  {"type":"latency", "first_response_ms": 780, …}
  {"type":"barge", "played_ms": 1234, "spoken_prefix": "您好，我是…"}
  {"type":"end", "closed": true, "slots": {…}, "writeback": {…}}
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import json
import os
import socketserver
import sys
import threading
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import providers as prov
from server.engine import Engine, load_json
from server.flow import State
from server.turn import EchoGate, TurnController

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

# 脚本台词模式下的"来电人"台词。多准备几句，避免一遍就念完然后干巴巴地"嗯"。
DEMO_SCRIPT = [
    "喂，你好，哪位？",
    "对，是我们公司",
    "嗯，我听明白了",
    "好的，我知道了，谢谢",
]
# 台词念完之后用的中性应答，让流程能靠"强制推进"继续走完，不会卡死
NEUTRAL_REPLIES = ["嗯，你继续说", "好的，你说", "可以", "行，我知道了"]


class ScriptedASR:
    """演示用识别器：按脚本依次返回用户台词。

    真实 ASR 接好后把 build_asr 里换成 HttpASR 即可，engine 不需要改。
    """

    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = list(lines or DEMO_SCRIPT)
        self.i = 0
        self.j = 0

    def next(self) -> str:
        if self.i < len(self.lines):
            t = self.lines[self.i]
            self.i += 1
            return t
        t = NEUTRAL_REPLIES[self.j % len(NEUTRAL_REPLIES)]
        self.j += 1
        return t


def build_asr(runtime: dict) -> tuple[Any, str]:
    """按配置选择 ASR 实现。返回 (asr, kind)。

    kind = "script" → 演示台词（默认，用于在没有密钥时验证链路）
    kind = "http"   → 任意 OpenAI 兼容的 /audio/transcriptions 接口，真实识别
    """
    cfg = runtime.get("providers", {}).get("asr", {})
    kind = (cfg.get("kind") or "script").lower()
    if kind == "http":
        hot_env = cfg.get("hotwords_env", "ASR_HOTWORDS")
        hot_raw = os.environ.get(hot_env, "") or ",".join(cfg.get("hotwords") or [])
        hotwords = [h.strip() for h in hot_raw.split(",") if h.strip()]
        asr = prov.HttpASR(
            base_url=cfg.get("base_url", ""),
            api_key=os.environ.get(cfg.get("api_key_env", "ASR_API_KEY"), ""),
            model=cfg.get("model", ""),
            language=cfg.get("language", "zh"),
            hotwords=hotwords,
        )
        return asr, "http"
    return ScriptedASR(), "script"


class CallSession:
    def __init__(self, ws: ServerConnection, runtime: dict) -> None:
        self.ws = ws
        self.runtime = runtime
        self.engine = Engine(runtime=runtime)
        self.asr, self.asr_kind = build_asr(runtime)
        self.echo = EchoGate()
        self.speaking = False
        self.last_turn: Any = None
        self.tc: TurnController | None = None
        # 本次话语的 PCM 缓冲，供"整句识别"用
        self.utt = bytearray()

    async def send(self, obj: dict) -> None:
        await self.ws.send(json.dumps(obj, ensure_ascii=False))

    async def run(self) -> None:
        await self.send({
            "type": "ready",
            "session": self.engine.session_id,
            "asr": self.asr_kind,
        })
        await self._speak_agent(await self.engine.opening())
        async for msg in self.ws:
            if isinstance(msg, (bytes, bytearray)):
                raw = bytes(msg)
                if self.utt is not None and len(self.utt) < prov.SAMPLE_RATE * 2 * 30:  # 最多缓冲 30s
                    self.utt.extend(raw)
                # 能量用于打断判据（客户端也会自己判一次，这里是服务端兜底）
                if self.speaking:
                    energy = prov.pcm16_rms(raw)
                    ok, why = self.echo.should_trigger(
                        now_ms=self.engine.log.t() * 1000,
                        speech_ms=200.0,
                        echo_score=min(0.99, energy * 4),
                    )
                    if ok:
                        await self.send({"type": "vad_hint", "reason": why})
                continue
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await self.send({"type": "error", "message": "bad_json"})
                continue
            await self._on_control(data)

    async def _on_control(self, data: dict) -> None:
        t = data.get("type")
        if t == "audio_end":
            await self._handle_utterance()
        elif t == "user_text":
            await self._handle_user(str(data.get("text", "")))
        elif t == "barge_in":
            await self._handle_barge(float(data.get("played_ms", 0)))
        elif t == "audio_start":
            self.utt = bytearray()
            await self.send({"type": "vad", "speaking": True})
        elif t == "script":
            self.asr = ScriptedASR(data.get("lines") or DEMO_SCRIPT)
            self.asr_kind = "script"
        elif t == "hangup":
            await self.send({
                "type": "end",
                "closed": self.engine.state in (State.CLOSED, State.CLOSE),
                "state": self.engine.state.value,
                "slots": self.engine.slots.flat(),
                "writeback": self.engine.writeback,
                "decision": self.engine.decision.to_dict() if self.engine.decision else None,
                "compliance": self.engine.compliance(),
            })

    async def _handle_utterance(self) -> None:
        """一句话说完：按配置的 ASR 模式取文本。"""
        if self.asr_kind == "script":
            await self._handle_user(self.asr.next())
            return

        pcm = bytes(self.utt)
        self.utt = bytearray()
        dur_ms = len(pcm) / 2 / prov.SAMPLE_RATE * 1000
        if not pcm or dur_ms < 200:
            await self.send({"type": "notice", "message": "没听清（音频太短），请再说一次"})
            return
        try:
            text = await self.asr.transcribe(pcm)
        except Exception as e:  # noqa: BLE001
            await self.send({"type": "error", "message": f"ASR 失败：{e}"})
            return
        if not text:
            await self.send({"type": "notice", "message": "没听清，请再说一次"})
            return
        await self.send({"type": "asr", "text": text, "dur_ms": round(dur_ms, 1)})
        await self._handle_user(text)

    async def _handle_user(self, text: str) -> None:
        if not text.strip():
            return
        await self.send({"type": "user", "text": text})
        if self.engine.state in (State.CLOSED, State.CLOSE, State.ESCALATE, State.HANDOFF):
            await self.send({"type": "end", "closed": True, "state": self.engine.state.value,
                             "slots": self.engine.slots.flat(), "writeback": self.engine.writeback,
                             "compliance": self.engine.compliance()})
            return
        turn = await self.engine.turn(text)
        await self._speak_agent(turn)
        if self.engine.state in (State.CLOSED, State.CLOSE, State.ESCALATE, State.HANDOFF):
            await self.send({
                "type": "end",
                "closed": self.engine.state in (State.CLOSED, State.CLOSE),
                "state": self.engine.state.value,
                "slots": self.engine.slots.flat(),
                "writeback": self.engine.writeback,
                "decision": self.engine.decision.to_dict() if self.engine.decision else None,
                "compliance": self.engine.compliance(),
            })

    async def _speak_agent(self, turn: Any) -> None:
        self.last_turn = turn
        reply = turn.reply
        # 用子句时长重建一轮播放记账，供打断时按"已播毫秒"裁剪
        self.tc = TurnController()
        for clause in prov._split_clause(reply):
            self.tc.on_tts_chunk(clause, prov.estimate_duration_ms(clause))
        self.tc.on_gen_done()

        await self.send({
            "type": "agent",
            "text": reply,
            "state": turn.state_after.value,
            "fallback": turn.used_fallback,
            "guard": turn.guard_tripped,
            "audio_ms": round(self.tc.generated_total_ms, 1),
        })
        await self.send({
            "type": "latency",
            "turn": turn.turn,
            "first_response_ms": round(turn.timeline.first_response_ms() or 0, 1),
            "llm_ttft_ms": round(turn.timeline.llm_ttft_ms() or 0, 1),
            "tts_ttfb_ms": round(turn.timeline.tts_ttfb_ms() or 0, 1),
        })
        self.speaking = True
        self.echo.set_agent_speaking(True, onset_ms=self.engine.log.t() * 1000)
        if turn.audio:
            # 按 100ms 一帧推：既保留流式播放，又不至于把消息条数打爆
            frame = prov.SAMPLE_RATE * 2 // 10
            for i in range(0, len(turn.audio), frame):
                await self.ws.send(turn.audio[i:i + frame])
        self.speaking = False
        self.echo.set_agent_speaking(False)

    async def _handle_barge(self, played_ms: float) -> None:
        if not self.tc:
            return
        self.tc.on_playout(played_ms)
        res = self.tc.barge_in()
        self.engine.log.emit("turn", "barge_in", played_ms=played_ms,
                             spoken=res.spoken_prefix, generated_ms=res.generated_total_ms)
        await self.send({
            "type": "barge",
            "played_ms": played_ms,
            "generated_ms": res.generated_total_ms,
            "spoken_prefix": res.spoken_prefix,
            "unspoken": res.unspoken,
        })


def start_static_server(directory: str, port: int) -> threading.Thread:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, directory=directory, **kw)

        def log_message(self, fmt: str, *args: Any) -> None:  # 别刷屏
            pass

    def run() -> None:
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
            httpd.serve_forever()

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def make_handler(runtime: dict):
    """返回一个可复用的连接处理器（测试里也用它，保证测的就是线上那份逻辑）。"""

    async def handler(ws: ServerConnection) -> None:
        session = CallSession(ws, runtime)
        try:
            await session.run()
        except Exception as e:  # noqa: BLE001
            try:
                await ws.send(json.dumps({"type": "error", "message": repr(e)}, ensure_ascii=False))
            except Exception:
                pass

    return handler


async def main_async(args: argparse.Namespace) -> None:
    runtime = load_json("config/runtime.json")
    if args.fast:
        runtime["latency_scale"] = 0.0

    start_static_server(WEB_DIR, args.web_port)
    print(f"[web] 打开 http://127.0.0.1:{args.web_port}/?ws={args.port}  （点【接通】）")
    print(f"[ws ] ws://127.0.0.1:{args.port}")

    async with serve(make_handler(runtime), "127.0.0.1", args.port, max_size=4 * 1024 * 1024):
        await asyncio.Future()  # 常驻


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765, help="WebSocket 端口")
    ap.add_argument("--web-port", type=int, default=8000, help="静态页面端口")
    ap.add_argument("--fast", action="store_true", help="去掉模拟时延，便于调试")
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[bye]")


if __name__ == "__main__":
    main()
