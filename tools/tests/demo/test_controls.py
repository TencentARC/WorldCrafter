import unittest
import numpy as np
from demo.camera import Action, Camera, ControlBuffer


class CameraTests(unittest.TestCase):
    def test_translation_directions_and_boundary(self):
        for action, axis, sign in [
            (Action(forward=1), 2, 1),
            (Action(forward=-1), 2, -1),
            (Action(right=1), 0, 1),
            (Action(right=-1), 0, -1),
            (Action(up=1), 1, -1),
            (Action(up=-1), 1, 1),
        ]:
            camera = Camera()
            local, world = camera.append(action)
            np.testing.assert_array_equal(local[0], np.eye(4, dtype=np.float32)[:3])
            self.assertAlmostEqual(float(local[-1, axis, 3]), sign * 32 / 33, places=6)
            local, world = camera.append(action)
            self.assertEqual(float(world[33, axis, 3]), sign)
            self.assertAlmostEqual(
                float(world[34, axis, 3] - world[32, axis, 3]), sign * 2 / 33, places=6
            )

    def test_rotation_and_local_global_consistency(self):
        camera = Camera()
        camera.append(Action(yaw=30))
        self.assertGreater(camera.world[0, 2], 0)
        start = camera.world.copy()
        local, world = camera.append(Action(forward=1))
        full = np.broadcast_to(np.eye(4), (33, 4, 4)).copy()
        full[:, :3] = local[-33:]
        np.testing.assert_allclose((start[None] @ full)[:, :3], world[-33:], atol=1e-7)
        self.assertGreater(world[-1, 0, 3], 0)
        for action, axis, sign in [
            (Action(yaw=-30), 0, -1),
            (Action(pitch=20), 1, -1),
            (Action(pitch=-20), 1, 1),
        ]:
            c = Camera()
            c.append(action)
            self.assertGreater(sign * c.world[axis, 2], 0)

    def test_elevation_stays_world_vertical_after_turn_and_pitch(self):
        camera = Camera()
        for a in [Action(yaw=27), Action(forward=1), Action(pitch=23)]:
            camera.append(a)
        prefix = [v.copy() for v in camera.poses()]
        for sign in (1, -1):
            start = camera.world.copy()
            local, world = camera.append(Action(up=sign, speed=0.6))
            full = np.broadcast_to(np.eye(4), (33, 4, 4)).copy()
            full[:, :3] = local[-33:]
            np.testing.assert_allclose(
                (start[None] @ full)[:, :3], world[-33:], atol=1e-7
            )
            np.testing.assert_array_equal(
                world[-33:, 0, 3], np.full(33, np.float32(start[0, 3]))
            )
            np.testing.assert_array_equal(
                world[-33:, 2, 3], np.full(33, np.float32(start[2, 3]))
            )
            self.assertAlmostEqual(camera.world[1, 3] - start[1, 3], -sign * 0.6)
            np.testing.assert_array_equal(local[:99], prefix[0])
            np.testing.assert_array_equal(world[:99], prefix[1])
            np.testing.assert_array_equal(local[-33], np.eye(4, dtype=np.float32)[:3])
        self.assertAlmostEqual(camera.world[1, 3], 0)

    def test_single_axis_required_and_limits(self):
        for action in [
            Action(forward=1, right=1),
            Action(up=1, yaw=2),
            Action(yaw=1, pitch=1),
        ]:
            with self.assertRaises(ValueError):
                action.normalized()
        self.assertEqual(Action(yaw=100).normalized().yaw, 30)
        self.assertEqual(Action(pitch=-100).normalized().pitch, -30)
        with self.assertRaises(ValueError):
            Action(up=float("nan")).normalized()


class ControlTests(unittest.TestCase):
    @staticmethod
    def key(c, key, down=True):
        c.update(dict(type="key", key=key, down=down))

    def test_all_ten_keys(self):
        expected = {
            "w": ("forward", 1),
            "s": ("forward", -1),
            "a": ("right", -1),
            "d": ("right", 1),
            "q": ("up", 1),
            "e": ("up", -1),
            "ArrowLeft": ("yaw", -15),
            "ArrowRight": ("yaw", 15),
            "ArrowUp": ("pitch", 15),
            "ArrowDown": ("pitch", -15),
        }
        for key, (axis, value) in expected.items():
            c = ControlBuffer()
            self.key(c, key)
            self.assertEqual(c.consume(), Action(**{axis: value}))

    def test_tap_hold_release_and_blur(self):
        c = ControlBuffer()
        self.key(c, "w")
        self.key(c, "w", False)
        self.assertEqual(c.consume().forward, 1)
        self.assertEqual(c.consume(), Action())
        self.key(c, "a")
        self.assertEqual(c.consume().right, -1)
        self.assertEqual(c.consume().right, -1)
        self.key(c, "a", False)
        self.assertEqual(c.consume(), Action())
        self.key(c, "q")
        c.update(dict(type="blur"))
        self.assertEqual(c.consume(), Action())

    def test_latest_tap_replaces_every_prior_category(self):
        c = ControlBuffer()
        for key in ("w", "q", "ArrowRight", "s", "e", "ArrowDown"):
            self.key(c, key)
            self.key(c, key, False)
            axis, sign = c.KEYS[key.lower()]
            value = sign * 15 if axis in ("yaw", "pitch") else sign
            self.assertEqual(c.peek(), Action(**{axis: value}))
        self.assertEqual(c.consume(), Action(pitch=-15))
        self.assertEqual(c.consume(), Action())

    def test_held_keys_do_not_combine_repeat_or_resurrect(self):
        c = ControlBuffer()
        self.key(c, "w")
        self.key(c, "q")
        self.key(c, "ArrowRight")
        self.key(c, "w")  # OS repeat of overridden key cannot steal priority.
        self.key(c, "q", False)  # Release of an old key cannot clear the latest.
        self.assertEqual(c.consume(), Action(yaw=15))
        self.assertEqual(c.consume(), Action(yaw=15))
        self.key(c, "ArrowRight", False)
        self.assertEqual(c.consume(), Action())  # W is still held, but overridden.
        self.key(c, "w", False)
        self.key(c, "w")  # A fresh physical press can select W again.
        self.assertEqual(c.consume(), Action(forward=1))

    def test_settings_edit_pending_but_never_consumed_action(self):
        c = ControlBuffer()
        c.update(dict(type="speed", value=1.5))
        c.update(dict(type="vertical_speed", value=0.6))
        self.key(c, "w")
        move = c.consume()
        self.key(c, "q")
        self.assertEqual(c.peek(), Action(up=1, speed=0.6))
        c.update(dict(type="vertical_speed", value=0.8))
        rise = c.consume()
        self.key(c, "ArrowUp")
        c.update(dict(type="rotation_angle", value=12))
        rotate = c.consume()
        c.update(dict(type="rotation_angle", value=20))
        self.assertEqual(move, Action(forward=1, speed=1.5))
        self.assertEqual(rise, Action(up=1, speed=0.8))
        self.assertEqual(rotate, Action(pitch=12, speed=1.5))
        self.assertEqual(c.peek(), Action(pitch=20, speed=1.5))

    def test_invalid_settings_and_removed_mouse(self):
        c = ControlBuffer()
        for kind in ("speed", "vertical_speed", "rotation_angle"):
            with self.assertRaises(ValueError):
                c.update(dict(type=kind, value=float("nan")))
            c.update(dict(type=kind, value=100))
            self.assertEqual(getattr(c, kind), 30 if kind == "rotation_angle" else 5)
        with self.assertRaises(ValueError):
            c.update(dict(type="mouse", yaw=1, pitch=2))


if __name__ == "__main__":
    unittest.main()
