"""Run with the pinned inference Python; tests raw prefix immutability, including non-anchors."""

import unittest
import torch
from worldcrafter.repencoder.trajectory_memory_provider import (
    RepEncoderInferenceMemoryProvider,
)


class AppendTests(unittest.TestCase):
    def test_strict_single_append_and_non_anchor_mutation(self):
        p = RepEncoderInferenceMemoryProvider(lambda **kw: None, append_only=True)
        poses = torch.eye(4).expand(1, 99, 4, 4).clone()
        p.append_trajectory(poses[:, :33], 0)
        bad = poses[:, :66].clone()
        bad[0, 1, 0, 3] = 1  # frame 1 is not a latent anchor
        with self.assertRaises(RuntimeError):
            p.append_trajectory(bad, 1)
        with self.assertRaises(ValueError):
            p.append_trajectory(poses, 2)
        p.append_trajectory(poses[:, :66], 1)
        p._initialize_or_validate_pose(pose=poses[:, :66], chunk_index=1)
        p.append_trajectory(poses, 2)
        p._initialize_or_validate_pose(pose=poses, chunk_index=2)
        self.assertEqual(p._global_pose_latents.shape, (27, 4, 4))
        p.reset_sequence()
        p.append_trajectory(poses[:, :33], 0)


if __name__ == "__main__":
    unittest.main()
