import tempfile
import unittest
from pathlib import Path

import numpy as np

from demo.camera import Camera, ControlBuffer
from worldcrafter.camera import (
    Action, action_from_event, build_trajectory, parse_trajectory,
    sample_chunk, save_trajectory,
)


class OrbitTests(unittest.TestCase):
    def test_zero_radius_matches_look(self):
        for axis in ("yaw", "pitch"):
            start = np.eye(4)
            expected = sample_chunk(start, Action(**{axis: 30}))
            actual = sample_chunk(start, Action(**{axis: 30}, orbit=True))
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)

    def test_radius_target_and_continuity(self):
        camera = Camera()
        target = np.array([0., 0., 2.])
        for action in (Action(yaw=20, orbit=True, orbit_radius=2),
                       Action(pitch=15, orbit=True, orbit_radius=2)):
            start = camera.world.copy()
            _, world = camera.append(action)
            poses = world[-33:].astype(np.float64)
            np.testing.assert_allclose(poses[:, :3, 3] + 2 * poses[:, :3, 2],
                                       np.broadcast_to(target, (33, 3)), atol=2e-7)
            np.testing.assert_allclose(poses[0], start[:3], atol=1e-7)

    def test_offline_and_interactive(self):
        events, options = parse_trajectory("forward1 orbit_left20x2 orbit_up15 right1")
        expected, _ = build_trajectory(events, orbit_radius=2)
        camera = Camera()
        for event in events:
            camera.append(action_from_event(event, 2))
        np.testing.assert_array_equal(camera.poses()[1], expected[:, :3].astype(np.float32))

    def test_reverse_and_export(self):
        events, options = parse_trajectory("orbit_left20x2 reverse2")
        options["orbit_radius"] = 2.
        poses, records = build_trajectory(events, **options)
        np.testing.assert_allclose(records[-1]["logical_end_c2w"], np.eye(4), atol=1e-14)
        with tempfile.TemporaryDirectory() as tmp:
            save_trajectory(Path(tmp), poses, records, events=events, options=options)
            restored, settings = parse_trajectory((Path(tmp) / "actions.txt").read_text())
            np.testing.assert_array_equal(build_trajectory(restored, **settings)[0], poses)

    def test_controls_and_limits(self):
        controls = ControlBuffer()
        controls.update(dict(type="rotation_mode", value="orbit"))
        controls.update(dict(type="orbit_radius", value=2))
        controls.update(dict(type="key", key="arrowleft", down=True))
        action = controls.consume()
        self.assertTrue(action.orbit)
        self.assertEqual(action.orbit_radius, 2)
        with self.assertRaises(ValueError):
            build_trajectory(["orbit_left180"], orbit_radius=5)


if __name__ == "__main__":
    unittest.main()
