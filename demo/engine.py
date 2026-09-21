"""All model calls, including generator advancement, run on the owning thread."""

import json
import time
import numpy as np
from PIL import Image
from .camera import Camera
from .config import DATA, MODEL_PATH, ROUTE
from .media import digest


class ChunkCancelled(Exception):
    pass


class WorldCrafterEngine:
    def __init__(self):
        import torch
        from worldcrafter import WorldCrafter

        self.torch = torch
        self.model = WorldCrafter.from_pretrained(
            MODEL_PATH,
            model_type="fast",
            device="cuda:0",
            attention_backend="native",
            enable_compile=True,
        )
        self.pipe = self.model.pipeline
        self.repencoder = self.model.memory_provider.runtime
        self.info = dict(self.model.fast_report, gpu_mode="single")
        self.devices = [0]
        self.session = None
        (DATA / "reports").mkdir(parents=True, exist_ok=True)
        (DATA / "reports/model_setup.json").write_text(
            json.dumps(self.info, indent=2) + "\n"
        )

    def initialize_session(self, image, prompt, seed, max_chunks, cancel):
        import torch
        from worldcrafter.repencoder import (
            RepEncoderInferenceMemoryProvider,
            RepEncoderInferenceProviderConfig,
        )
        from worldcrafter.streaming import stream_chunks

        self.close_session()
        self.cancel = cancel
        self.provider = RepEncoderInferenceMemoryProvider(
            self.repencoder,
            RepEncoderInferenceProviderConfig(
                seed=seed,
                trajectory_fov_horizontal_fov_degrees=100.0,
                trajectory_fov_vertical_fov_degrees=71.13349068444832,
            ),
            append_only=True,
        )
        self.camera = Camera()
        self.index = 0
        self.pipe.stage_model_trace.clear()
        if hasattr(self.pipe, "resident_branches"):
            self.pipe.resident_branches.switch("equal")
        self.generator = torch.Generator(device="cuda:0").manual_seed(seed)
        self.session = stream_chunks(
            self.pipe,
            image=Image.open(image).convert("RGB"),
            prompt=prompt,
            generator=self.generator,
            provider=self.provider,
            max_chunks=max_chunks,
            cancel=self.check_cancel,
            on_step=self.step_completed,
        )
        for device in self.devices:
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            ready = next(self.session)
            if ready != {"ready": True}:
                raise RuntimeError("The inference session did not initialize")
        self.check_cancel()

    def check_cancel(self):
        if self.cancel.is_set():
            raise ChunkCancelled()

    def step_completed(self, pipe, step, timestep, callback_kwargs):
        # Host-side telemetry only: never change tensors or synchronize the GPU.
        callback = getattr(self, "on_step_completed", None)
        if callback is not None:
            callback()
        return callback_kwargs

    def next_chunk(self, action, poses=None, capture=False):
        torch = self.torch
        start = time.perf_counter()
        self.check_cancel()
        local, world = self.camera.append(action) if poses is None else poses
        camera = dict(
            pose=torch.from_numpy(local[None]).to("cuda"),
            retrieval_pose=torch.from_numpy(world[None]).to("cuda"),
            x_fov=torch.tensor([100.0], device="cuda"),
            xi=torch.tensor([0.0], device="cuda"),
        )
        with torch.inference_mode():
            self.provider.append_trajectory(camera["retrieval_pose"], self.index)
            result = self.session.send(camera)
        for device in self.devices:
            torch.cuda.synchronize(device)
        result["generation_s"] = time.perf_counter() - start
        if capture:
            result["validation_tensors"] = {
                key: result[key].clone() for key in ("latents", "rgb", "rng_state")
            }
        # NumPy has no BF16 dtype; hash the original bytes without rounding.
        result["latent_sha256"] = digest(
            result.pop("latents").contiguous().view(torch.uint8).numpy()
        )
        result["rgb_sha256"] = digest(
            result["rgb"].contiguous().view(torch.uint8).numpy()
        )
        result["rng_sha256"] = digest(
            result.pop("rng_state").contiguous().view(torch.uint8).numpy()
        )
        rgb = self.pipe.video_processor.postprocess_video(
            result.pop("rgb"), output_type="np"
        )[0]
        result["frames"] = (rgb * 255).astype(np.uint8)
        result["memory"] = self.memory()
        result["route"] = self.pipe.stage_model_trace[-6:]
        if [x["branch"] for x in result["route"]] != [
            "equal" if b == "A" else "old" for b in ROUTE
        ]:
            raise RuntimeError("Unexpected I2V model routing")
        result["retrieval"] = (
            None if self.index == 0 else self.provider.render_records[-1].to_jsonable()
        )
        result["local_pose"] = local[-33:]
        result["global_pose"] = world[-33:]
        self.index += 1
        return result

    def memory(self):
        t = self.torch.cuda
        devices = [
            dict(
                device=d,
                name=t.get_device_name(d),
                allocated=t.memory_allocated(d),
                reserved=t.memory_reserved(d),
                peak_allocated=t.max_memory_allocated(d),
                peak_reserved=t.max_memory_reserved(d),
            )
            for d in self.devices
        ]
        return dict(
            **{
                key: sum(d[key] for d in devices)
                for key in ("allocated", "reserved", "peak_allocated", "peak_reserved")
            },
            devices=devices,
        )

    def close_session(self):
        if self.session is not None:
            with self.torch.inference_mode():
                self.session.close()
            self.session = None
        if hasattr(self, "provider"):
            self.provider.reset_sequence()


class MockEngine:
    """Deterministic CPU renderer for protocol, browser, and lifecycle tests."""

    def __init__(self, **kwargs):
        self.session = None
        self.info = {"mode": "mock"}

    def initialize_session(self, image, prompt, seed, max_chunks, cancel):
        self.cancel = cancel
        self.camera = Camera()
        self.index = 0
        self.image = np.asarray(
            Image.open(image).convert("RGB").resize((640, 384))
        ).copy()
        self.seed = seed
        self.session = True

    def next_chunk(self, action, poses=None):
        start = time.perf_counter()
        for _ in range(6):
            if self.cancel.wait(0.06):
                raise ChunkCancelled()
            callback = getattr(self, "on_step_completed", None)
            if callback is not None:
                callback()
        local, world = self.camera.append(action) if poses is None else poses
        frames = np.repeat(self.image[None], 33, axis=0)
        for j in range(33):
            x = int((self.index * 33 + j + self.seed) % 630)
            frames[j, :10, x : x + 10] = (90, 255, 180)
        result = dict(
            chunk_index=self.index,
            frames=frames,
            generation_s=time.perf_counter() - start,
            latent_sha256="mock",
            rgb_sha256=digest(frames),
            rng_sha256="mock",
            memory=self.memory(),
            route=[
                dict(
                    chunk=self.index,
                    stage=j // 2,
                    step=j % 2,
                    branch="equal" if ROUTE[j] == "A" else "old",
                    finite=True,
                )
                for j in range(6)
            ],
            retrieval=None if self.index == 0 else {"chunk_index": self.index},
            local_pose=local[-33:],
            global_pose=world[-33:],
        )
        self.index += 1
        return result

    def memory(self):
        return dict(allocated=0, reserved=0, peak_allocated=0, peak_reserved=0)

    def close_session(self):
        self.session = None
