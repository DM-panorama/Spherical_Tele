"""CPU regressions for the TeleStyle-compatible A1 pilot."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from omegaconf import OmegaConf
from PIL import Image

from training.checkpoint import validate_adapter_metadata
from training.conditioning import (
    cache_style_latent, split_style_paths, style_cache_key,
)
from training.models import configured_lora_paths
from training.train_sphere_adapter import (
    _only_adapter_trainable,
    _set_lightning_schedule,
)
from tests.test_sphere_adapter import SphereAdapterTests
from training.geometry import build_sphere_geometry


class _FourStepScheduler:
    def set_timesteps(self, count, **kwargs):
        self.count = count
        self.kwargs = kwargs
        self.timesteps = [torch.tensor(value) for value in (900, 700, 400, 100)]
        self.sigmas = [torch.tensor(value) for value in (0.9, 0.7, 0.4, 0.1, 0.0)]


class _FakeVae:
    def __init__(self):
        self.encode_calls = 0

    def encode(self, pixels, **kwargs):
        self.encode_calls += 1
        return torch.full((1, 2, 2, 2), float(self.encode_calls))


class _FakePipe:
    device = torch.device("cpu")
    torch_dtype = torch.float32

    def __init__(self):
        self.vae = _FakeVae()

    def preprocess_image(self, image):
        return torch.zeros(1, 3, image.height, image.width)

    def load_models_to_device(self, names):
        self.loaded = names

class TeleStylePilotTests(unittest.TestCase):
    def test_lora_config_is_optional_and_rejects_duplicates(self):
        empty = OmegaConf.create({"model": {}})
        self.assertEqual(configured_lora_paths(empty, "configs/a1.yaml"), [])
        duplicate = OmegaConf.create(
            {"model": {"lora_paths": ["weights/a.safetensors"] * 2}}
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            configured_lora_paths(duplicate, "configs/a1.yaml")

    def test_style_split_holds_out_style_four_and_cache_key_uses_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in range(1, 5):
                path = root / f"style_{index}.jpg"
                path.write_bytes(f"style-{index}".encode())
                paths.append(path)
            training, validation = split_style_paths(paths)
            self.assertEqual([path.stem for path in training],
                             ["style_1", "style_2", "style_3"])
            self.assertEqual(validation.stem, "style_4")
            self.assertNotEqual(
                style_cache_key(paths[0], 512), style_cache_key(paths[0], 1024)
            )
    def test_style_latent_cache_reuses_sha_and_size_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "style_1.jpg"
            Image.new("RGB", (8, 12), "blue").save(source)
            pipe = _FakePipe()
            first = cache_style_latent(pipe, source, root / "cache", 16)
            second = cache_style_latent(pipe, source, root / "cache", 16)
            torch.testing.assert_close(first, second)
            self.assertEqual(pipe.vae.encode_calls, 1)
            self.assertEqual(len(list((root / "cache").glob("*.pt"))), 1)

    def test_four_step_schedule_uses_only_lightning_points(self):
        scheduler = _FourStepScheduler()
        timesteps, sigmas = _set_lightning_schedule(scheduler, 1024)
        self.assertEqual(scheduler.count, 4)
        self.assertEqual(len(timesteps), 4)
        self.assertEqual(len(sigmas), 4)
        self.assertEqual(scheduler.kwargs["dynamic_shift_len"], 4096)

    def test_teacher_bypasses_position_and_cross_adapter_with_two_edits(self):
        fixture = SphereAdapterTests()
        model = fixture._model()
        geometry = build_sphere_geometry(2, 15, 10, 2, torch.device("cpu"))
        latent = torch.randn(1, 1, 4, 4)
        style = torch.randn(1, 1, 2, 2)
        prompt = torch.randn(1, 3, 8)
        mask = torch.ones(1, 3, dtype=torch.long)
        with (
            mock.patch.object(
                model.adapter, "add_position",
                side_effect=AssertionError("position adapter was called"),
            ),
            mock.patch.object(
                model.adapter, "forward",
                side_effect=AssertionError("cross adapter was called"),
            ),
        ):
            outputs = model(
                latent, latent, [latent, style], [latent, style],
                prompt, prompt, mask, mask, torch.tensor([500.0]), geometry,
                adapter_enabled=False,
            )
        self.assertEqual(outputs[0].shape, latent.shape)
        self.assertTrue(_only_adapter_trainable(model))

    def test_new_checkpoint_metadata_validates_and_legacy_is_compatible(self):
        validate_adapter_metadata(
            {"chart_size": 1024},
            base_transformer_sha256="anything",
            lora_sha256={"anything": "anything"},
        )
        metadata = {
            "schema_version": 2,
            "base_transformer_index_sha256": "base",
            "lora_sha256": {"style": "one", "lightning": "two"},
            "prompt": "prompt",
            "lightning_timesteps": [900.0, 700.0, 400.0, 100.0],
            "south_yaw_degrees": 180.0,
        }
        validate_adapter_metadata(
            metadata,
            base_transformer_sha256="base",
            lora_sha256={"style": "one", "lightning": "two"},
            prompt="prompt",
            lightning_timesteps=[900.0, 700.0, 400.0, 100.0],
            south_yaw_degrees=180.0,
        )
        with self.assertRaisesRegex(ValueError, "lora_sha256"):
            validate_adapter_metadata(
                metadata, lora_sha256={"style": "changed"}
            )


if __name__ == "__main__":
    unittest.main()
