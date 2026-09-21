import hashlib
import io
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageOps


def image_input(data):
    with Image.open(io.BytesIO(data)) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        return ImageOps.fit(
            image, (640, 384), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
        )


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def encode_chunk(frames, path):
    path = Path(path)
    temp = path.with_suffix(".part.mp4")
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.shape != (33, 384, 640, 3):
        raise ValueError(f"Expected 33 RGB frames at 640x384, got {frames.shape}")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "640x384",
            "-framerate",
            "16",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-threads",
            "2",
            str(temp),
        ],
        input=frames.tobytes(),
        check=True,
    )
    temp.replace(path)


def concat_recording(directory):
    directory = Path(directory)
    chunks = sorted(directory.glob("chunk_*.mp4"))
    if not chunks:
        return None
    listing = directory / "concat.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in chunks))
    # Immutable filename: another download must not replace a FileResponse in flight.
    target = directory / f"session_{len(chunks):03d}_chunks.mp4"
    if target.exists():
        return target
    temp = target.with_suffix(".part.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            str(listing),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(temp),
        ],
        check=True,
    )
    temp.replace(target)
    return target
