#!/usr/bin/env python3
"""Real chunks: compile, pause/resume, translation, orbit, and video decoding."""
import asyncio
import json
import os
import time
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import websockets

PORT = int(os.environ.get("WORLDCRAFTER_SMOKE_PORT", "8080"))
GPUS = int(os.environ.get("WORLDCRAFTER_SMOKE_GPUS", "1"))
URL = f"http://127.0.0.1:{PORT}"


def request(path, data=None):
    req = Request(
        URL + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=10) as response:
        return json.load(response)


async def main():
    deadline = time.monotonic() + 1200
    while True:
        try:
            health = request("/api/health")
            if health["error"]:
                raise RuntimeError(health["error"])
            if health["ready"]:
                break
        except OSError:
            pass
        assert time.monotonic() < deadline, "Model startup timeout"
        await asyncio.sleep(2)
    assert health["gpu_mode"] == ("single" if GPUS == 1 else "query_parallel")
    assert health["compile_enabled"]
    assert health["route"] == ["A"] * 5 + ["B"]
    presets = request("/api/presets")
    preset = next(p for p in presets["presets"] if p["id"] == presets["default"])
    body = dict(preset=preset["id"], prompt=preset["prompt"], seed=42, max_chunks=3)
    created = request("/api/sessions", body)
    sid = created["session_id"]
    try:
        request("/api/sessions", body)
        raise AssertionError("Second active session accepted")
    except HTTPError as exc:
        assert exc.code == 409
    events = []
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws/{sid}") as ws:

        async def send(kind, **values):
            await ws.send(json.dumps(dict(type=kind, generation=0, **values)))

        await send("key", key="w", down=True)
        while True:
            event = json.loads(await asyncio.wait_for(ws.recv(), 1200))
            events.append(event)
            assert event.get("type") != "error", event
            assert not event.get("error"), event
            if event["type"] == "chunk_started" and event["chunk_index"] == 0:
                await send("pause")
                await send("vertical_speed", value=0.6)
                await send("key", key="w", down=True)
                await send("key", key="q", down=True)
                await send("key", key="q", down=False)
            if event["type"] == "status" and event["state"] == "paused":
                assert event["pending"]["up"] == 1 and event["pending"]["forward"] == 0
                assert event["pending"]["speed"] == 0.6
                await send("resume")
            if event["type"] == "chunk_started" and event["chunk_index"] == 1:
                assert event["action"]["up"] == 1 and event["action"]["forward"] == 0
                await send("rotation_mode", value="orbit")
                await send("orbit_radius", value=2)
                await send("key", key="arrowleft", down=True)
                await send("key", key="arrowleft", down=False)
            if event["type"] == "chunk_started" and event["chunk_index"] == 2:
                assert event["action"]["orbit"] and event["action"]["orbit_radius"] == 2
            if event["type"] == "chunk_ready":
                assert len(event["memory"]["devices"]) == GPUS
                assert [s["branch"] for s in event["route"]] == ["equal"] * 5 + ["old"]
            if event["type"] == "status" and event["state"] == "complete":
                break
    status = request(f"/api/sessions/{sid}")
    with urlopen(URL + f"/api/sessions/{sid}/recording?generation=0") as response:
        video = response.read()
    out = Path(
        os.environ.get(
            "WORLDCRAFTER_SMOKE_OUTPUT",
            str(Path(__file__).resolve().parents[3] / "output/demo-smoke"),
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "session.mp4").write_bytes(video)
    (out / "report.json").write_text(
        json.dumps(
            dict(
                health=health,
                session_id=sid,
                single_session_rejected=True,
                latest_override=True,
                pause_resume=True,
                status=status,
                events=events,
            ),
            indent=2,
        )
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(out / "session.mp4"),
            "-f",
            "null",
            "-",
        ],
        check=True,
    )
    frames = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames,width,height",
                "-of",
                "json",
                str(out / "session.mp4"),
            ]
        )
    )["streams"][0]
    assert (frames["nb_read_frames"], frames["width"], frames["height"]) == (
        "99",
        640,
        384,
    )
    print("DEMO_SERVER_PASS", sid, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
