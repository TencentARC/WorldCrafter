import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from PIL import Image
import numpy as np
import torch

from demo.engine import WorldCrafterEngine
from worldcrafter.cli import parse_args
from worldcrafter.fast.compact_ucpe import compact_ucpe
from worldcrafter.inference import WorldCrafter
from worldcrafter.output import save_chunk_state, sha256


class InferenceIOTests(unittest.TestCase):
    def test_default_outputs_are_distinct_and_explicit_path_is_preserved(self):
        paths = [
            parse_args(["--model-type", model, "--mode", mode]).output_path
            for model in ("base", "fast")
            for mode in ("i2v", "t2v")
            for _ in range(2)
        ]
        self.assertEqual(len(set(paths)), 8)
        self.assertTrue(
            all(
                path.name == "video.mp4" and path.parents[3].name == "output"
                for path in paths
            )
        )
        self.assertEqual(
            parse_args(["--output-path", "chosen.mp4"]).output_path, Path("chosen.mp4")
        )

    def test_chunk_callback_requires_output_directory_before_inference(self):
        model = WorldCrafter.__new__(WorldCrafter)
        with self.assertRaisesRegex(ValueError, "on_chunk_saved requires"):
            model.generate(
                mode="i2v",
                camera_path=Path("missing.npy"),
                output_path=Path("unused.mp4"),
                prompt="scene",
                negative_prompt="",
                on_chunk_saved=lambda index, path: None,
            )

    def test_session_initialization_advances_generator(self):
        engine = WorldCrafterEngine.__new__(WorldCrafterEngine)
        engine.torch = torch
        engine.device = torch.device("cuda:0")
        engine.session = None
        engine.repencoder = object()
        engine.pipe = types.SimpleNamespace(stage_model_trace=[])
        engine.devices = []
        advanced = []

        def stream(*args, **kwargs):
            advanced.append(True)
            yield {"ready": True}

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            Image.new("RGB", (8, 8)).save(image)
            with patch("worldcrafter.streaming.stream_chunks", stream), patch(
                "worldcrafter.repencoder.RepEncoderInferenceMemoryProvider"
            ), patch("torch.Generator"):
                engine.initialize_session(image, "scene", 42, 1, threading.Event())
                self.assertEqual(advanced, [True])
                engine.close_session()

    def test_ucpe_compaction_rejects_lossy_storage(self):
        layer = torch.nn.Linear(2, 2)
        model = types.SimpleNamespace(
            blocks=[types.SimpleNamespace(cam_self_attn=layer)]
        )
        with torch.no_grad():
            layer.weight.fill_(0.123456)
            layer.bias.zero_()
        with self.assertRaisesRegex(ValueError, "lose information"):
            compact_ucpe(model)
        with torch.no_grad():
            layer.weight.copy_(layer.weight.bfloat16().float())
        x = torch.ones(1, 2)
        expected = layer(x).detach()
        compact_ucpe(model)
        self.assertTrue(torch.equal(layer(x), expected))

    def test_generation_only_enables_requested_output_callbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            camera = directory / "camera.npy"
            np.save(camera, np.tile(np.eye(4), (33, 1, 1)))
            calls = []
            frames = np.zeros((33, 4, 4, 3), dtype=np.float32)

            class Pipeline:
                video_processor = types.SimpleNamespace(
                    postprocess_video=lambda video, output_type: [frames]
                )

                def __call__(self, **kwargs):
                    calls.append(kwargs)
                    if kwargs["callback_on_chunk_end"] is not None:
                        kwargs["callback_on_chunk_end"](0, None)
                    if kwargs["callback_on_chunk_state"] is not None:
                        kwargs["callback_on_chunk_state"](
                            0,
                            dict(
                                format="worldcrafter_chunk_state_v1",
                                completed_chunk_index=0,
                                next_chunk_index=1,
                            ),
                        )
                    return types.SimpleNamespace(frames=[frames])

            provider = types.SimpleNamespace(
                reset_sequence=lambda: None,
                render_records=[],
                runtime=types.SimpleNamespace(report={"model_sha256": "model"}),
            )
            model = WorldCrafter(
                pipeline=Pipeline(),
                memory_provider=provider,
                model_path=directory,
                device=torch.device("cpu"),
                attention_backend="native",
                adapter_load={},
                height=4,
                width=4,
            )
            args = dict(
                mode="t2v",
                camera_path=camera,
                output_path=directory / "video.mp4",
                prompt="scene",
                negative_prompt="",
            )
            with patch(
                "worldcrafter.inference.export_to_video",
                lambda frames, path, fps: Path(path).write_bytes(b"video"),
            ):
                model.generate(**args)
                self.assertIsNone(calls[-1]["callback_on_chunk_end"])
                self.assertIsNone(calls[-1]["callback_on_chunk_state"])
                saved = []
                model.generate(
                    **args,
                    chunk_output_dir=directory / "chunks",
                    state_output_dir=directory / "states",
                    on_chunk_saved=lambda index, path: saved.append(path)
                )
                self.assertEqual(saved, [directory / "chunks/chunk_000_33f.mp4"])
                self.assertTrue((directory / "states/latest.json").is_file())
                with self.assertRaisesRegex(RuntimeError, "reference mismatch"):

                    def reject(index, path):
                        raise RuntimeError("reference mismatch")

                    model.generate(
                        **args,
                        chunk_output_dir=directory / "chunks",
                        on_chunk_saved=reject
                    )

    def test_resume_state_keeps_contract_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            state = dict(
                format="worldcrafter",
                completed_chunk_index=0,
                next_chunk_index=1,
                latents=torch.arange(8),
            )
            contract = {"seed": 42}
            history = [{"chunk": 0}]
            save_chunk_state(
                0,
                state,
                state_output_dir=directory,
                run_contract=contract,
                history_selection=history,
            )
            path = directory / "chunk_000_complete.pt"
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved["run_contract"], contract)
            self.assertEqual(saved["history_selection"], history)
            self.assertTrue(torch.equal(saved["latents"], state["latents"]))
            metadata = json.loads((directory / "latest.json").read_text())
            self.assertEqual(metadata["checkpoint_sha256"], sha256(path))
            self.assertFalse(list(directory.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
