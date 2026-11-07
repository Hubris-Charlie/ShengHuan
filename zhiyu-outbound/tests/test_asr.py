"""ASR 适配器测试：用一个本地假服务验证 WAV 打包、请求格式与响应解析。

这条链路是「麦克风里的话 → 文字」的唯一通道，所以必须用真实的 HTTP 往返来验，
而不是只断言函数返回值。
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import providers as prov
from server.ws_server import DEMO_SCRIPT, NEUTRAL_REPLIES, ScriptedASR, build_asr

RECEIVED: dict = {}


class FakeASRHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静音
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        RECEIVED["path"] = self.path
        RECEIVED["auth"] = self.headers.get("Authorization", "")
        RECEIVED["content_type"] = self.headers.get("Content-Type", "")
        RECEIVED["body"] = body
        payload = json.dumps({"text": "我是，你说"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class TestAsrAdapter(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeASRHandler)
        cls.port = cls.httpd.server_address[1]
        cls.th = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.th.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()

    def test_wav_packaging(self) -> None:
        pcm = b"\x01\x00" * 16000  # 1 秒静音的形状
        wav = prov.pcm16_to_wav(pcm, 16000)
        self.assertTrue(wav.startswith(b"RIFF"))
        with wave.open(io.BytesIO(wav), "rb") as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.getframerate(), 16000)
            self.assertEqual(w.getnframes(), 16000)

    async def test_http_asr_roundtrip(self) -> None:
        pcm = b"\x10\x00" * 8000
        asr = prov.HttpASR(
            base_url=f"http://127.0.0.1:{self.port}/v1",
            api_key="test-key",
            model="test-model",
            language="zh",
            hotwords=["安心服务权益", "上门检测"],
        )
        text = await asr.transcribe(pcm)
        self.assertEqual(text, "我是，你说")
        # 请求确实发出去了，且带上了该带的东西
        self.assertTrue(RECEIVED["path"].endswith("/v1/audio/transcriptions"))
        self.assertEqual(RECEIVED["auth"], "Bearer test-key")
        self.assertIn("multipart/form-data", RECEIVED["content_type"])
        self.assertIn(b"test-model", RECEIVED["body"])
        self.assertIn(b"audio/wav", RECEIVED["body"])
        # 热词必须以 prompt 形式带进去（这是领域词表定制的落点）
        self.assertIn("安心服务权益".encode("utf-8"), RECEIVED["body"])

    async def test_http_asr_empty_pcm(self) -> None:
        asr = prov.HttpASR(base_url="http://127.0.0.1:1/v1", api_key="k")
        self.assertEqual(await asr.transcribe(b""), "")

    async def test_http_asr_failure_propagates(self) -> None:
        """接口不可达时必须抛出来，让上层决定"请再说一次"，而不是静默返回空。"""
        asr = prov.HttpASR(base_url="http://127.0.0.1:1/v1", api_key="k", timeout=2)
        with self.assertRaises(Exception):
            await asr.transcribe(b"\x00\x00" * 100)


class TestAsrSelection(unittest.TestCase):
    def test_script_mode(self) -> None:
        rt = {"providers": {"asr": {"kind": "script"}}}
        asr, kind = build_asr(rt)
        self.assertEqual(kind, "script")
        self.assertIsInstance(asr, ScriptedASR)
        # 按脚本顺序返回，念完后走中性应答而不是卡在同一个词上
        self.assertEqual(asr.next(), DEMO_SCRIPT[0])
        for _ in range(len(DEMO_SCRIPT) - 1):
            asr.next()
        self.assertIn(asr.next(), NEUTRAL_REPLIES)

    def test_http_mode(self) -> None:
        rt = {"providers": {"asr": {"kind": "http", "base_url": "https://x/v1",
                                    "model": "m", "api_key_env": "ASR_API_KEY",
                                    "hotwords_env": "ASR_HOTWORDS"}}}
        os.environ["ASR_HOTWORDS"] = "甲,乙"
        try:
            asr, kind = build_asr(rt)
            self.assertEqual(kind, "http")
            self.assertIsInstance(asr, prov.HttpASR)
            self.assertEqual(asr.hotwords, ["甲", "乙"])
        finally:
            os.environ.pop("ASR_HOTWORDS", None)

    def test_default_is_script(self) -> None:
        asr, kind = build_asr({})
        self.assertEqual(kind, "script")


if __name__ == "__main__":
    unittest.main(verbosity=2)
