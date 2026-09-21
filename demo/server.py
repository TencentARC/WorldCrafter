import asyncio
from contextlib import asynccontextmanager
import hashlib
import os
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .config import ROOT, DATA, DEFAULT_PRESET, presets, ROUTE
from .media import image_input, concat_recording
from .service import Manager


@asynccontextmanager
async def lifespan(app):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "uploads").mkdir(exist_ok=True)
    app.state.manager = Manager(
        asyncio.get_running_loop(), mock=os.environ.get("WORLDCRAFTER_DEMO_MOCK") == "1"
    )
    yield
    app.state.manager.close()


app = FastAPI(lifespan=lifespan)
DATA.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
app.mount("/inputs", StaticFiles(directory=ROOT / "inputs"), name="inputs")
app.mount("/sessions", StaticFiles(directory=DATA), name="sessions")


@app.get("/")
def index():
    return FileResponse(ROOT / "static/index.html")


@app.get("/api/health")
def health():
    m = app.state.manager
    return dict(
        ready=m.ready,
        error=m.error,
        mock=m.mock,
        route=list(ROUTE),
        pyramid=[2, 2, 2],
        fps=16,
        compile_enabled=m.model_info.get("compile_enabled", False),
        gpu_mode="single",
    )


@app.get("/api/presets")
def get_presets():
    return dict(default=DEFAULT_PRESET, presets=presets())


@app.post("/api/upload")
async def upload(request: Request):
    data = bytearray()
    async for part in request.stream():
        data.extend(part)
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(413, "Image exceeds 20 MB")
    try:
        image = await asyncio.to_thread(image_input, bytes(data))
    except Exception as exc:
        raise HTTPException(400, f"Invalid image: {exc}")
    key = hashlib.sha256(image.tobytes()).hexdigest()
    path = DATA / "uploads" / f"{key}.png"
    image.save(path)
    return dict(
        upload_id=key, preview=f"/sessions/uploads/{key}.png", width=640, height=384
    )


class CreateSession(BaseModel):
    preset: str = DEFAULT_PRESET
    upload_id: str | None = Field(None, pattern=r"^[a-f0-9]{64}$")
    prompt: str = Field(min_length=1, max_length=10000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    max_chunks: int = Field(default=50, ge=1, le=50)


@app.post("/api/sessions")
def create_session(body: CreateSession):
    m = app.state.manager
    if m.error:
        raise HTTPException(503, m.error)
    if body.preset not in {p["id"] for p in presets()}:
        raise HTTPException(400, "Unknown preset")
    image = (
        DATA / "uploads" / f"{body.upload_id}.png"
        if body.upload_id
        else ROOT / "inputs" / body.preset / "input.png"
    )
    if not image.is_file():
        raise HTTPException(404, "Input image is missing")
    try:
        return m.create(image, body.prompt, body.seed, body.max_chunks)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/sessions/{sid}")
def status(sid: str):
    m = app.state.manager
    with m.cv:
        s = m.sessions.get(sid)
        if s is None:
            raise HTTPException(404, "Unknown session")
        return dict(**s.snapshot(), completed_chunks=list(s.results))


@app.get("/api/sessions/{sid}/recording")
async def recording(sid: str, generation: int | None = None):
    m = app.state.manager
    with m.cv:
        s = m.sessions.get(sid)
        if s is None:
            raise HTTPException(404, "Unknown session")
        g = s.generation if generation is None else generation
        if not 0 <= g <= s.generation:
            raise HTTPException(404, "Unknown generation")
        directory = DATA / sid / f"g{g:03d}"
    # Snapshot a completed prefix. Published chunks and recordings are immutable.
    if not hasattr(app.state, "recording_lock"):
        app.state.recording_lock = asyncio.Lock()
    async with app.state.recording_lock:
        path = await asyncio.to_thread(concat_recording, directory)
    if path is None:
        raise HTTPException(409, "No completed chunk yet")
    return FileResponse(
        path, media_type="video/mp4", filename=f"worldcrafter_{sid[:8]}_g{g}.mp4"
    )


@app.websocket("/ws/{sid}")
async def websocket(ws: WebSocket, sid: str):
    m = app.state.manager
    queue = asyncio.Queue(maxsize=128)
    await ws.accept()
    try:
        m.connect(sid, queue)
    except (KeyError, ValueError) as exc:
        await ws.send_json(dict(type="error", message=str(exc)))
        await ws.close(code=1008)
        return

    async def sender():
        while True:
            await ws.send_json(await queue.get())

    task = asyncio.create_task(sender())
    try:
        while True:
            msg = await ws.receive_json()
            if not isinstance(msg, dict):
                raise ValueError("Message must be an object")
            try:
                m.command(sid, msg)
            except (ValueError, TypeError, KeyError) as exc:
                await ws.send_json(dict(type="error", message=str(exc)))
    except (WebSocketDisconnect, ValueError):
        pass
    finally:
        task.cancel()
        m.disconnect(sid, queue)
