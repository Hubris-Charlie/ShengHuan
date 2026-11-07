"""端到端通路测试：真实 WebSocket + 真实音频字节 + 真实打断上报。

这一层测的是"浏览器 ↔ 服务端"的协议与播放记账，不是业务逻辑
（业务逻辑在 test_core.py 里已经覆盖）。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    HAVE_WS = True
except Exception:  # pragma: no cover
    HAVE_WS = False

from server.engine import load_json
from server.ws_server import make_handler


@unittest.skipUnless(HAVE_WS, "需要 websockets：pip install websockets")
class TestWsCall(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.runtime = load_json("config/runtime.json")
        self.runtime["latency_scale"] = 0.0     # 测试不睡觉
        self.audio_frames = 0

    async def recv_json(self, ws, want: str, limit: int = 3000) -> dict:
        for _ in range(limit):
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            if isinstance(msg, (bytes, bytearray)):
                self.audio_frames += 1
                continue
            m = json.loads(msg)
            if m.get("type") == want:
                return m
        raise AssertionError(f"等不到消息类型 {want}")

    async def drain(self, ws, want_frames: int = 1, limit: int = 3000) -> None:
        """把消息流里的二进制音频帧收满（音频是在 latency 之后才开始推的）。"""
        for _ in range(limit):
            if self.audio_frames >= want_frames:
                return
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            if isinstance(msg, (bytes, bytearray)):
                self.audio_frames += 1

    async def test_full_protocol(self) -> None:
        async with serve(make_handler(self.runtime), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                self.assertEqual(ready["type"], "ready")
                self.assertTrue(ready["session"])

                # --- 开场 ---
                opening = await self.recv_json(ws, "agent")
                # 话术是轮转的，"我是 / 这边是 / 这里是" 都算自报身份，不能写死一种
                self.assertTrue(
                    any(k in opening["text"] for k in ("我是", "这边是", "这里是")),
                    f"开场没有自报身份：{opening['text']}",
                )
                lat0 = await self.recv_json(ws, "latency")
                self.assertIsNotNone(lat0["first_response_ms"])
                await self.drain(ws, want_frames=1)
                self.assertGreater(self.audio_frames, 0, "服务端没推音频帧")

                # --- 一轮用户输入 ---
                await ws.send(json.dumps({"type": "user_text", "text": "我是，你说"}, ensure_ascii=False))
                u = await self.recv_json(ws, "user")
                self.assertEqual(u["text"], "我是，你说")
                a1 = await self.recv_json(ws, "agent")
                self.assertEqual(a1["state"], "REJECT_REASON")
                lat1 = await self.recv_json(ws, "latency")
                self.assertIsNotNone(lat1["first_response_ms"])

                # --- 打断：上报真实已播毫秒 ---
                await ws.send(json.dumps({"type": "barge_in", "played_ms": 600}))
                b = await self.recv_json(ws, "barge")
                self.assertEqual(b["played_ms"], 600)
                self.assertGreater(b["generated_ms"], 0)
                # 已播 600ms，必然只听到前半句，不可能等于全句
                self.assertLessEqual(len(b["spoken_prefix"]), len(a1["text"]))
                self.assertTrue(
                    len(b["spoken_prefix"]) < len(a1["text"]) or b["generated_ms"] <= 600,
                    "已播 600ms 却报告说完了整句，说明按生成进度而不是播放进度算了",
                )

                # --- 挂断 ---
                await ws.send(json.dumps({"type": "hangup"}))
                end = await self.recv_json(ws, "end")
                self.assertIn("state", end)
                self.assertIn("compliance", end)
                self.assertIn("mandatory_missing", end["compliance"])

    async def next_of(self, ws, kinds: set[str], limit: int = 3000) -> tuple[str, dict]:
        """读到下一个属于 kinds 的消息（跳过音频帧）。"""
        for _ in range(limit):
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            if isinstance(msg, (bytes, bytearray)):
                self.audio_frames += 1
                continue
            m = json.loads(msg)
            if m.get("type") in kinds:
                return m["type"], m
        raise AssertionError(f"等不到 {kinds} 中的任何一种消息")

    async def test_scripted_asr_drives_full_call(self) -> None:
        """走服务端脚本化的"识别结果"，把整通电话跑完。"""
        async with serve(make_handler(self.runtime), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                await asyncio.wait_for(ws.recv(), timeout=5)   # ready
                await self.recv_json(ws, "agent")              # opening

                got_end: dict | None = None
                for _ in range(8):
                    await ws.send(json.dumps({"type": "audio_end"}))
                    kind, m = await self.next_of(ws, {"agent", "end"})
                    if kind == "end":
                        got_end = m
                        break
                    if m.get("state") in ("CLOSED", "CLOSE", "ESCALATE", "HANDOFF"):
                        got_end = await self.recv_json(ws, "end")
                        break

                self.assertIsNotNone(got_end, "整通电话没有走到结束")
                self.assertIn("compliance", got_end)
                self.assertIsInstance(got_end["compliance"]["mandatory_missing"], list)
                self.assertEqual(got_end["compliance"]["mandatory_missing"], [],
                                 "正常走完的电话不应缺必说话术")


if __name__ == "__main__":
    unittest.main(verbosity=2)
