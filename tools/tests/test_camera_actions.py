import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from demo.camera import Camera
from worldcrafter.camera import (
    action_from_event, build_trajectory, count_chunks, parse_actions, parse_trajectory, save_trajectory,
)
from worldcrafter.cli import DEFAULT_I2V_CAMERA, DEFAULT_T2V_CAMERA, parse_args, prepare_camera


class CameraActionTests(unittest.TestCase):
    def test_names_repetitions_comments_and_precision(self):
        short = parse_actions("f1x2, yl30x3 # turn\nb0.123456789123456789")
        full = parse_actions("forward1 forward1 yaw_left30x3 backward0.123456789123456789")
        self.assertEqual(short, full)
        camera, _ = build_trajectory(short)
        self.assertEqual(camera.shape, (198, 4, 4))
        self.assertEqual(camera.dtype, np.float64)
        self.assertEqual(short[-1], "backward0.123456789123456789")

    def test_invalid_actions_and_translation_limit(self):
        for text in ("", "# empty", "forward", "f-1", "orbit30", "f1x0", "f1x-2", "f1x2.5", "up5.1", "f6"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_actions(text)
        build_trajectory(parse_actions("forward5 up5 backward5 down5"))

    def test_default_trajectory_prefix_is_unchanged(self):
        camera, _ = build_trajectory(parse_actions("f1x2 b1x4 yr45x2 yl45x4 r1x2 yr45x4 yl45"))
        original = np.load(DEFAULT_I2V_CAMERA)
        if original.ndim == 4:
            original = original[0]
        np.testing.assert_array_equal(camera[:, :3], original[:len(camera), :3])

    def test_demo_and_script_share_all_ten_actions(self):
        events = parse_actions("yr25 pu20 f1 b0.5 l0.7 r0.3 up0.4 down0.2 yl15 pd10")
        expected, _ = build_trajectory(events)
        demo = Camera()
        for event in events:
            demo.append(action_from_event(event))
        local, global_pose = demo.poses()
        np.testing.assert_array_equal(global_pose, expected[:, :3].astype(np.float32))
        for index in range(len(events)):
            offset = index * 33
            full_local = np.tile(np.eye(4), (33, 1, 1))
            full_local[:, :3] = local[offset:offset + 33]
            np.testing.assert_allclose(
                (expected[offset] @ full_local)[:, :3],
                global_pose[offset:offset + 33], atol=1e-7,
            )

    def test_pitch_does_not_change_horizontal_movement_height(self):
        for pitch in ("pu30", "pd30", "pu90", "pd90"):
            camera, records = build_trajectory(parse_actions(f"{pitch} f1 r1 up1"))
            self.assertTrue(np.isfinite(camera).all())
            np.testing.assert_array_equal(camera[33:99, 1, 3], 0)
            end = np.array(records[2]["logical_end_c2w"])
            np.testing.assert_allclose(end[:3, 3], [1, 0, 1], atol=1e-12)
            np.testing.assert_array_equal(camera[99:, 0, 3], 1)
            np.testing.assert_array_equal(camera[99:, 2, 3], 1)
        camera, records = build_trajectory(parse_actions("pu30 yr30"))
        np.testing.assert_allclose(
            camera[33, 1, 2], np.array(records[-1]["logical_end_c2w"])[1, 2], atol=1e-12,
        )

    def test_cli_sources_and_saved_trajectory_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "actions.txt"
            source.write_text("\ufeff# path\nf1x2\nyl30\n", encoding="utf-8")
            expected, records = build_trajectory(parse_actions("f1x2 yl30"))
            standalone = save_trajectory(root / "standalone", expected, records)
            for model in ("base", "fast"):
                for mode in ("i2v", "t2v"):
                    for selection in (["--actions", "f1x2 yl30"], ["--actions-file", str(source)]):
                        args = parse_args([
                            "--model-type", model, "--mode", mode,
                            "--output-path", str(root / f"{model}_{mode}.mp4"), *selection,
                        ])
                        prepare_camera(args)
                        self.assertEqual(args.camera_path.read_bytes(), standalone.read_bytes())
                        saved = (args.camera_path.parent / "actions.txt").read_text()
                        rebuilt, _ = build_trajectory(parse_actions(saved))
                        np.testing.assert_array_equal(rebuilt, expected)
            args = parse_args(["--actions-file", str(source), "--num-chunks", "2"])
            self.assertEqual(len(args.camera_events), 3)
            self.assertEqual(args.num_chunks, 2)
            with self.assertRaises(ValueError):
                parse_args(["--actions-file", str(source), "--num-chunks", "4"])
            for selection in (
                ["--actions", "f1", "--camera-path", str(standalone)],
                ["--actions", "f1", "--actions-file", str(source)],
            ):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(selection)
            with self.assertRaises(ValueError):
                parse_args(["--actions", "f1", "--local-camera-path", str(standalone)])

    def test_file_and_default_inputs_remain_unchanged(self):
        for mode, default in (("i2v", DEFAULT_I2V_CAMERA), ("t2v", DEFAULT_T2V_CAMERA)):
            args = parse_args(["--mode", mode])
            prepare_camera(args)
            self.assertEqual(args.camera_path, default)
        args = parse_args(["--camera-path", "custom.npy"])
        prepare_camera(args)
        self.assertEqual(args.camera_path, Path("custom.npy"))

    def test_compound_and_reverse_actions(self):
        events = parse_actions("forward2 & right2 & yaw_left45 reverse1")
        poses, records = build_trajectory(events)
        self.assertEqual(count_chunks(events), 2)
        np.testing.assert_allclose(records[-1]["logical_end_c2w"], np.eye(4), atol=1e-14)
        np.testing.assert_allclose(poses[1:33], poses[34:66][::-1], atol=1e-14)
        with self.assertRaises(ValueError):
            parse_actions("forward1&backward1")
        with self.assertRaises(ValueError):
            build_trajectory(parse_actions("forward4&right4"))
        with self.assertRaises(ValueError):
            build_trajectory(parse_actions("forward1 reverse2"))
        poses, _ = build_trajectory(parse_actions("forward1x2 reverse2"), last_frame="include")
        np.testing.assert_array_equal(poses[-1], np.eye(4))

    def test_reverse_frames_and_sampling_headers(self):
        events, options = parse_trajectory("@dtype float32\n@sampling smooth_turns\nf1x2 yl30x2 reverse_frames4")
        camera, _ = build_trajectory(events, **options)
        self.assertEqual(count_chunks(events), 8)
        self.assertEqual(camera.dtype, np.float32)
        np.testing.assert_array_equal(camera[132:], camera[:132][::-1])
        with self.assertRaises(ValueError):
            parse_trajectory("@sampling unknown\nf1")
        with self.assertRaises(ValueError):
            parse_trajectory("f1\n@dtype float32")

    def test_bundled_cases_and_action_round_trip(self):
        root = Path(__file__).resolve().parents[2] / "test"
        cases = sorted(root.glob("*/*/actions.txt"))
        self.assertGreaterEqual(len(cases), 10)
        for path in cases:
            with self.subTest(case=path.parent), tempfile.TemporaryDirectory() as temp:
                events, options = parse_trajectory(path.read_text())
                camera, records = build_trajectory(events, **options)
                reference = np.load(path.with_name("camera.npy"))
                self.assertEqual(camera.dtype, reference.dtype)
                np.testing.assert_array_equal(camera[:, :3], reference)
                saved = save_trajectory(Path(temp), camera, records, events=events, options=options)
                again, settings = parse_trajectory(saved.with_name("actions.txt").read_text())
                rebuilt, _ = build_trajectory(again, **settings)
                np.testing.assert_array_equal(rebuilt, camera)
                args = parse_args(["--actions-file", str(path), "--num-chunks", "1"])
                self.assertEqual(count_chunks(args.camera_events), len(camera) // 33)


if __name__ == "__main__":
    unittest.main()
