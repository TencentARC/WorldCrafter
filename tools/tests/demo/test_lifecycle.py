import asyncio
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from PIL import Image
from demo import service


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data_patch = patch.object(service, "DATA", self.root)
        self.data_patch.start()
        self.encoder = patch.object(
            service,
            "encode_chunk",
            lambda frames, path: Path(path).write_bytes(b"test"),
        )
        self.encoder.start()
        self.image = self.root / "input.png"
        Image.new("RGB", (640, 384), "#345637").save(self.image)
        self.m = service.Manager(asyncio.get_running_loop(), mock=True)
        await self.until(lambda: self.m.ready)

    async def asyncTearDown(self):
        self.m.close()
        await asyncio.to_thread(self.m.thread.join, 5)
        self.encoder.stop()
        self.data_patch.stop()
        self.temp.cleanup()

    async def until(self, predicate, timeout=5):
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.02)

    def create(self, action=True):
        info = self.m.create(self.image, "test scene", 42, 5)
        s = self.m.sessions[info["session_id"]]
        q = asyncio.Queue()
        self.m.connect(s.id, q)
        if action:
            self.tap(s)
        return s, q

    def cmd(self, s, kind, **extra):
        self.m.command(s.id, dict(type=kind, generation=s.generation, **extra))

    def tap(self, s, key="w"):
        self.cmd(s, "key", key=key, down=True)
        self.cmd(s, "key", key=key, down=False)

    async def test_idle_requires_action_and_tap_runs_once(self):
        s, q = self.create(action=False)
        await self.until(lambda: s.state == "waiting_action")
        self.cmd(s, "speed", value=2)
        await asyncio.sleep(0.5)
        self.assertEqual(s.chunks, 0)
        self.assertIsNone(s.active)
        self.tap(s)
        await self.until(lambda: s.chunks == 1 and s.state == "waiting_action")
        await asyncio.sleep(0.5)
        self.assertEqual(s.chunks, 1)
        self.cmd(s, "key", key="q", down=True)
        await self.until(lambda: s.chunks >= 2)
        self.cmd(s, "blur")
        await self.until(lambda: s.state == "waiting_action")
        count = s.chunks
        await asyncio.sleep(0.5)
        self.assertEqual(s.chunks, count)
        self.cmd(s, "reset")
        await self.until(lambda: s.state == "waiting_action")
        self.assertEqual(s.chunks, 0)

    async def test_six_step_progress_and_reset(self):
        info = self.m.create(self.image, "test scene", 42, 1)
        s = self.m.sessions[info["session_id"]]
        q = asyncio.Queue()
        self.m.connect(s.id, q)
        for generation in (0, 1):
            self.tap(s)
            values = []
            async with asyncio.timeout(5):
                while True:
                    event = await q.get()
                    if event.get("type") != "status" or event["generation"] != generation:
                        continue
                    values.append(event["completed_steps"])
                    if event["state"] == "complete":
                        break
            self.assertEqual(sorted(set(values)), list(range(7)))
            self.assertEqual(values, sorted(values))
            self.assertEqual(s.completed_steps, 6)
            if generation == 0:
                self.cmd(s, "reset")
                self.assertEqual(s.completed_steps, 0)

    async def test_pause_resume_tap_and_reset_determinism(self):
        s, q = self.create()
        await self.until(lambda: s.active is not None)
        self.cmd(s, "pause")
        await self.until(lambda: s.state == "paused")
        self.assertEqual(s.chunks, 1)
        first = s.results[0]["rgb_sha256"]
        self.cmd(s, "key", key="w", down=True)
        self.cmd(s, "key", key="w", down=False)
        self.assertEqual(s.controls.peek().forward, 1)
        await asyncio.sleep(0.2)
        self.assertEqual(s.chunks, 1)
        self.cmd(s, "resume")
        await self.until(lambda: s.active is not None)
        self.assertEqual(s.active["forward"], 1)
        self.cmd(s, "pause")
        await self.until(lambda: s.state == "paused")
        self.cmd(s, "reset")
        self.tap(s)
        await self.until(lambda: s.active is not None)
        self.cmd(s, "pause")
        await self.until(lambda: s.state == "paused")
        self.assertEqual(s.generation, 1)
        self.assertEqual(s.chunks, 1)
        self.assertEqual(s.results[0]["rgb_sha256"], first)
        self.m.command(s.id, dict(type="key", key="w", down=True, generation=0))
        self.assertEqual(s.controls.peek().forward, 0)

    async def test_single_session_disconnect_and_cancelled_generation(self):
        s, q = self.create()
        await self.until(lambda: s.active is not None)
        with self.assertRaises(ValueError):
            self.m.create(self.image, "second", 42, 1)
        self.cmd(s, "reset")
        self.tap(s)
        await self.until(lambda: s.active is not None)
        self.cmd(s, "pause")
        await self.until(lambda: s.state == "paused")
        self.assertEqual(s.chunks, 1)
        self.assertEqual(s.generation, 1)
        self.assertFalse((self.root / s.id / "g000/chunk_000.mp4").exists())
        self.tap(s)
        self.cmd(s, "resume")
        await self.until(lambda: s.active is not None)
        self.m.disconnect(s.id, q)
        await asyncio.sleep(0.6)
        self.assertEqual(s.chunks, 1)
        self.assertTrue(s.cancel.is_set())
        self.assertEqual(s.state, "disconnected")
        self.m.create(self.image, "new", 42, 1)

    async def test_latest_input_until_chunk_start_and_active_is_immutable(self):
        s, q = self.create()
        await self.until(lambda: s.active is not None)
        self.cmd(s, "pause")
        await self.until(lambda: s.state == "paused")
        for key in ("w", "q", "ArrowRight"):
            self.cmd(s, "key", key=key, down=True)
            self.cmd(s, "key", key=key, down=False)
        self.cmd(s, "rotation_angle", value=12)
        self.assertEqual(s.controls.peek().yaw, 12)
        self.cmd(s, "resume")
        await self.until(lambda: s.active is not None)
        active = s.active.copy()
        self.assertEqual(active["yaw"], 12)
        self.assertEqual(active["forward"], 0)
        self.assertEqual(active["up"], 0)
        for key in ("a", "q", "e"):
            self.cmd(s, "key", key=key, down=True)
            self.cmd(s, "key", key=key, down=False)
        self.cmd(s, "vertical_speed", value=0.6)
        self.cmd(s, "rotation_angle", value=20)
        self.assertEqual(s.active, active)
        self.cmd(s, "pause")
        await self.until(lambda: s.state == "paused")
        self.cmd(s, "resume")
        await self.until(lambda: s.active is not None)
        self.assertEqual(s.active["up"], -1)
        self.assertEqual(s.active["speed"], 0.6)
        self.assertEqual(s.active["yaw"], 0)
        self.assertEqual(s.active["right"], 0)
        self.cmd(s, "pause")
        await self.until(lambda: s.state == "paused")
        self.assertEqual(s.results[1]["action"], active)
        self.assertEqual(s.results[2]["action"]["up"], -1)

    async def test_stop_during_encode_never_publishes_partial_chunk(self):
        encoding = threading.Event()
        release = threading.Event()

        def slow_encode(frames, path):
            encoding.set()
            release.wait(3)
            Path(path).write_bytes(b"uncommitted")

        self.encoder.stop()
        self.encoder = patch.object(service, "encode_chunk", slow_encode)
        self.encoder.start()
        s, q = self.create()
        await self.until(encoding.is_set)
        self.cmd(s, "stop")
        release.set()
        await self.until(lambda: s.active is None)
        self.assertEqual(s.chunks, 0)
        self.assertEqual(list(s.directory.glob("chunk_*.mp4")), [])
        self.assertEqual(list(s.directory.glob("pending_*.mp4")), [])


if __name__ == "__main__":
    unittest.main()
