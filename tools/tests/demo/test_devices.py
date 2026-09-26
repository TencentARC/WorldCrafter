import os
import sys
import unittest
from unittest.mock import patch

from demo.__main__ import main


class DeviceSelectionTests(unittest.TestCase):
    def test_device_selection_respects_visible_gpus(self):
        for indices, expected in [("0", "2"), ("1", "5"), ("0,1", "2,5")]:
            with (
                self.subTest(indices=indices),
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,5"}),
                patch.object(sys, "argv", ["demo", "--devices", indices]),
                patch("uvicorn.run"),
            ):
                main()
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], expected)
                self.assertEqual(os.environ["WORLDCRAFTER_DEMO_GPUS"], str(len(indices.split(","))))


if __name__ == "__main__":
    unittest.main()
