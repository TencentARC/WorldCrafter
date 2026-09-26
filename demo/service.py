"""Bounded controls and one model owner, independent from HTTP and playback."""

from dataclasses import dataclass, field
import json
import threading
import time
import traceback
import uuid
import numpy as np
from .camera import ControlBuffer
from .config import DATA, ROUTE
from .engine import ChunkCancelled, WorldCrafterEngine, MockEngine
from .media import encode_chunk


def create_engine(mock=False):
    if mock:
        return MockEngine()
    import os
    if os.environ.get("WORLDCRAFTER_DEMO_GPUS") == "2":
        from .parallel import ParallelEngine
        return ParallelEngine()
    return WorldCrafterEngine()


@dataclass
class Session:
    id: str
    image: str
    prompt: str
    seed: int
    max_chunks: int
    generation: int = 0
    chunks: int = 0
    completed_steps: int = 0
    state: str = "waiting_connection"
    connected: bool = False
    paused: bool = False
    stopped: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)
    controls: ControlBuffer = field(default_factory=ControlBuffer)
    active: dict | None = None
    started_at: float = field(default_factory=time.time)
    local_poses: list = field(default_factory=list)
    global_poses: list = field(default_factory=list)
    results: list = field(default_factory=list)
    events: list = field(default_factory=list)
    error: str | None = None

    @property
    def directory(self):
        return DATA / self.id / f"g{self.generation:03d}"

    def snapshot(self):
        last = self.results[-1] if self.results else {}
        return dict(
            type="status",
            session_id=self.id,
            generation=self.generation,
            state=self.state,
            chunks=self.chunks,
            completed_steps=self.completed_steps,
            max_chunks=self.max_chunks,
            active=self.active,
            pending=self.controls.peek().json(),
            control_settings=dict(
                speed=self.controls.speed,
                vertical_speed=self.controls.vertical_speed,
                rotation_angle=self.controls.rotation_angle,
                rotation_mode=self.controls.rotation_mode,
                orbit_radius=self.controls.orbit_radius,
            ),
            generation_s=last.get("generation_s"),
            generation_fps=last.get("generation_fps"),
            memory=last.get("memory"),
            error=self.error,
            connected=self.connected,
        )


class Manager:
    def __init__(self, loop, mock=False):
        self.loop = loop
        self.mock = mock
        self.cv = threading.Condition()
        self.sessions = {}
        self.active = None
        self.listeners = {}
        self.ready = False
        self.model_info = {}
        self.error = None
        self.shutdown = False
        self.thread = threading.Thread(
            target=self.run, name="worldcrafter-gpu-owner", daemon=True
        )
        self.thread.start()

    def emit(self, session, event=None):
        payload = dict(event or session.snapshot())
        payload.update(session_id=session.id, generation=session.generation)

        def deliver():
            queue = self.listeners.get(session.id)
            if queue:
                if queue.full():
                    # Preserve every playable chunk (at most 50). Coalesce telemetry
                    # and discard old generations instead of losing video descriptors.
                    completed = []
                    while not queue.empty():
                        previous = queue.get_nowait()
                        if (
                            previous.get("type") == "chunk_ready"
                            and previous.get("generation") == payload["generation"]
                        ):
                            completed.append(previous)
                    for previous in completed:
                        queue.put_nowait(previous)
                queue.put_nowait(payload)

        self.loop.call_soon_threadsafe(deliver)

    def create(self, image, prompt, seed, max_chunks):
        with self.cv:
            if (
                self.active
                and not self.active.stopped
                and self.active.state not in ("complete", "error")
            ):
                raise ValueError("Only one active session is allowed")
            s = Session(uuid.uuid4().hex, str(image), prompt, seed, max_chunks)
            self.sessions[s.id] = s
            self.active = s
            self.persist(s)
            self.cv.notify_all()
            return s.snapshot()

    def persist(self, s):
        s.directory.mkdir(parents=True, exist_ok=True)
        payload = dict(
            session_id=s.id,
            generation=s.generation,
            image=s.image,
            prompt=s.prompt,
            seed=s.seed,
            max_chunks=s.max_chunks,
            state=s.state,
            route=list(ROUTE),
            fps=16,
            chunks=s.results,
            controls=s.events,
        )
        path = s.directory / "session.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2) + "\n")
        temp.replace(path)
        if s.local_poses:
            np.save(
                s.directory / "pose_local_model_input.npy",
                np.concatenate(s.local_poses),
            )
            np.save(
                s.directory / "pose_global_retrieval.npy",
                np.concatenate(s.global_poses),
            )

    def connect(self, sid, queue):
        with self.cv:
            s = self.sessions[sid]
            if sid in self.listeners:
                raise ValueError("This session already has a controller")
            if s is not self.active or s.stopped:
                raise ValueError("Session has stopped; create a new session")
            self.listeners[sid] = queue
            s.connected = True
            s.state = "queued" if self.ready else "loading_model"
            self.emit(s)
            self.cv.notify_all()

    def disconnect(self, sid, queue):
        with self.cv:
            if self.listeners.get(sid) is not queue:
                return
            self.listeners.pop(sid, None)
            s = self.sessions[sid]
            s.connected = False
            s.controls.clear()
            s.stopped = True
            s.cancel.set()
            s.state = "disconnected"
            self.persist(s)
            self.cv.notify_all()

    def command(self, sid, message):
        with self.cv:
            s = self.sessions[sid]
            if message.get("generation") != s.generation:
                return
            if s is not self.active or s.stopped:
                raise ValueError("Session is no longer active")
            kind = message.get("type")
            if kind in ("key", "speed", "vertical_speed", "rotation_angle", "rotation_mode", "orbit_radius", "blur"):
                s.controls.update(message)
            elif kind == "playback_started":
                if not isinstance(message.get("chunk_index"), int):
                    raise ValueError("Invalid chunk index")
            elif kind == "pause":
                s.paused = True
                s.state = "pausing" if s.active else "paused"
            elif kind == "resume":
                s.paused = False
                s.state = "generating" if s.active else "queued"
            elif kind == "stop":
                s.stopped = True
                s.cancel.set()
                s.controls.clear()
                s.state = "stopped"
            elif kind == "reset":
                s.cancel.set()
                self.persist(s)
                s.generation += 1
                s.cancel = threading.Event()
                s.chunks = 0
                s.completed_steps = 0
                s.controls.clear()
                s.paused = False
                s.state = "queued"
                s.active = None
                s.started_at = time.time()
                s.local_poses = []
                s.global_poses = []
                s.results = []
                s.events = []
                self.persist(s)
                self.emit(s, dict(type="reset"))
            else:
                raise ValueError("Unknown command")
            s.events.append(dict(received_at=time.time(), chunk=s.chunks, **message))
            self.emit(s)
            self.cv.notify_all()

    def run(self):
        engine = None
        owned = None
        try:
            engine = create_engine(self.mock)
            self.model_info = getattr(engine, "info", {})
            with self.cv:
                self.ready = True
                if self.active:
                    self.emit(self.active)
                self.cv.notify_all()
            while True:
                with self.cv:
                    if self.shutdown:
                        break
                    s = self.active
                    key = None if s is None else (s.id, s.generation)
                    needs_close = owned is not None and (
                        owned != key or s.stopped or s.chunks >= s.max_chunks
                    )
                    eligible = (
                        s is not None
                        and s.connected
                        and not s.stopped
                        and not s.paused
                        and s.chunks < s.max_chunks
                    )
                    if eligible and not s.controls.has_action():
                        if s.state != "waiting_action":
                            s.state = "waiting_action"
                            self.emit(s)
                        eligible = False
                    if not needs_close and not eligible:
                        self.cv.wait(timeout=1)
                        continue
                    generation, cancel = (
                        (None, None) if s is None else (s.generation, s.cancel)
                    )
                if needs_close:
                    engine.close_session()
                    owned = None
                    continue
                try:
                    if owned != key:
                        with self.cv:
                            s.state = "initializing"
                            self.emit(s)
                        start = time.perf_counter()
                        engine.initialize_session(
                            s.image, s.prompt, s.seed, s.max_chunks, cancel
                        )
                        init_s = time.perf_counter() - start
                        owned = key
                    with self.cv:
                        if generation != s.generation or cancel.is_set() or s.paused:
                            continue
                        if not s.controls.has_action():
                            continue
                        action = s.controls.consume()
                        s.active = action.json()
                        s.completed_steps = 0
                        s.state = "generating"
                        accepted_at = time.time()
                        index = s.chunks
                        directory = s.directory
                        self.emit(
                            s,
                            dict(
                                type="chunk_started",
                                chunk_index=index,
                                action=s.active,
                                accepted_at=accepted_at,
                            ),
                        )
                        self.emit(s)

                    def step_completed():
                        with self.cv:
                            # A reset or stop may arrive while a forward is running.
                            if generation != s.generation or cancel.is_set():
                                return
                            s.completed_steps = min(6, s.completed_steps + 1)
                            self.emit(s)

                    engine.on_step_completed = step_completed
                    try:
                        result = engine.next_chunk(action)
                    finally:
                        engine.on_step_completed = None
                    if cancel.is_set():
                        raise ChunkCancelled()
                    encode_start = time.perf_counter()
                    path = directory / f"chunk_{index:03d}.mp4"
                    pending_path = directory / f"pending_{index:03d}.mp4"
                    encode_chunk(result.pop("frames"), pending_path)
                    result.update(
                        encode_s=time.perf_counter() - encode_start,
                        generation_fps=33 / result["generation_s"],
                        action=action.json(),
                        accepted_at=accepted_at,
                        ready_at=time.time(),
                        initialization_s=init_s if index == 0 else 0.0,
                        url=f"/sessions/{s.id}/g{generation:03d}/{path.name}",
                    )
                    with self.cv:
                        if generation != s.generation or cancel.is_set():
                            pending_path.unlink(missing_ok=True)
                            if generation == s.generation:
                                s.active = None
                                self.emit(s)
                            continue
                        pending_path.replace(path)
                        s.local_poses.append(result.pop("local_pose"))
                        s.global_poses.append(result.pop("global_pose"))
                        s.results.append(result)
                        s.chunks += 1
                        s.active = None
                        s.state = (
                            "complete"
                            if s.chunks >= s.max_chunks
                            else ("paused" if s.paused else "queued")
                        )
                        self.persist(s)
                        self.emit(s, dict(type="chunk_ready", **result))
                        self.emit(s)
                except ChunkCancelled:
                    engine.close_session()
                    owned = None
                    with self.cv:
                        if generation == s.generation:
                            s.active = None
                            self.emit(s)
                except Exception as exc:
                    traceback.print_exc()
                    engine.close_session()
                    owned = None
                    with self.cv:
                        if generation == s.generation:
                            s.error = str(exc)
                            s.state = "error"
                            s.stopped = True
                            s.active = None
                            self.persist(s)
                            self.emit(s)
        except Exception as exc:
            traceback.print_exc()
            with self.cv:
                self.error = str(exc)
                if self.active:
                    self.active.error = self.error
                    self.active.state = "error"
                    self.emit(self.active)
        finally:
            if engine:
                if hasattr(engine, "shutdown"):
                    engine.shutdown()
                else:
                    engine.close_session()

    def close(self):
        with self.cv:
            self.shutdown = True
            if self.active:
                self.active.cancel.set()
            self.cv.notify_all()
        # Cancellation is handled by the model owner at a denoising boundary.
        self.thread.join(timeout=330)
        if self.thread.is_alive():
            raise RuntimeError(
                "GPU owner did not shut down within the collective timeout"
            )
