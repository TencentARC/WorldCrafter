"""Local checkpoint layout and shared-component resolution."""
import json
from pathlib import Path
import tempfile
import unittest

from worldcrafter.cli import parse_args
from worldcrafter.inference import validate_weights


class WeightLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / 'WorldCrafter-Base'
        self.fast = self.root / 'WorldCrafter-Fast'
        for name in ('transformer', 'adapter'):
            (self.base / name).mkdir(parents=True)
        for name in ('camera_adapter.pth', 'pytorch_lora_weights.safetensors'):
            (self.base / 'adapter' / name).touch()

    def shared(self, root):
        for name in ('repencoder', 'text_encoder', 'tokenizer', 'vae', 'scheduler'):
            (root / name).mkdir(parents=True)
        for name in ('model.safetensors', 'config.json', 'manifest.json'):
            (root / 'repencoder' / name).touch()

    def test_base_uses_fast_components_and_its_own_adapter(self):
        self.shared(self.fast)
        (self.base / 'inference_config.json').write_text(json.dumps({
            'format': 'worldcrafter_base_v1', 'shared_components': '../WorldCrafter-Fast',
        }))
        paths = validate_weights(self.base)
        self.assertEqual(paths['adapter'], self.base / 'adapter')
        self.assertEqual(paths['transformer'], self.base / 'transformer')
        for name in ('repencoder', 'text_encoder', 'tokenizer', 'vae', 'scheduler'):
            self.assertEqual(paths[name], self.fast / name)
        (self.fast / 'repencoder' / 'model.safetensors').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'WorldCrafter-Fast/repencoder/model'):
            validate_weights(self.base)

    def test_legacy_self_contained_base(self):
        self.shared(self.base)
        self.assertEqual(validate_weights(self.base)['repencoder'], self.base / 'repencoder')

    def test_cli_defaults(self):
        self.assertEqual(parse_args([]).model_path.name, 'WorldCrafter-Base')
        self.assertEqual(parse_args(['--model-type', 'fast']).model_path.name, 'WorldCrafter-Fast')


if __name__ == '__main__':
    unittest.main()
