# WorldCrafter Interactive Demo

Explore a scene from an image with keyboard camera controls, powered by
WorldCrafter-Fast. Compilation is enabled, so the first generation takes longer.

## Run

Validated on a single NVIDIA H200.
Place the complete `WorldCrafter-Fast` folder under `weights/` as described in
the main README. Base weights are not required for the demo.
Install FFmpeg so that `ffmpeg` and `ffprobe` are available on `PATH`.

Follow the [environment setup](../README.md#2-environment), activate
your uv or conda environment, then run from the repository root:

```bash
python -m demo \
  --model-path weights/WorldCrafter-Fast \
  --output-dir output/demo
```

Open `http://localhost:8080`. Select a preset or upload an image, edit its prompt,
and start a session. One session can generate at a time. Outputs include video
chunks, camera trajectories, and session metadata in the output directory.

The service listens on `127.0.0.1` by default. For a remote machine, forward the
port with `ssh -N -L 8080:127.0.0.1:8080 <host>`. Use `--host` and `--port` to
change the listen address. Run one server process per GPU; multiple web workers
would each load a separate model. This demo has no authentication, so use an
authenticated proxy when making it publicly accessible.

## Controls

| Input | Action |
| --- | --- |
| W / S | `forward` / `backward` on the horizontal plane |
| A / D | `left` / `right` on the horizontal plane |
| Q / E | Move up / down along the world vertical axis |
| Left / right arrows | `yaw_left` / `yaw_right`: turn around the world vertical axis |
| Up / down arrows | `pitch_up` / `pitch_down`: look up / down |
| Esc or window blur | Clear pending movement |

Each chunk uses one action. A new key replaces the pending action until the next
chunk starts. Tap for one chunk or hold for continued movement. With no pending
action, generation waits. Pause, resume, restart, and video download are
available in the page.

The demo and [script camera actions](../README.md#4-camera-actions) use the same
motion rules. Looking up or down does not change the height of W/S/A/D movement.
