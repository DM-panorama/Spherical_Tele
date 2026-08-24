import unittest
from unittest import mock

import torch
from PIL import Image

import telestyleimage_inference as image_inference
import telestylepanorama_inference as panorama_inference


class _FakeScheduler:
    def set_timesteps(self, steps, **kwargs):
        self.timesteps = [torch.tensor(index) for index in range(steps)]


class _FakeVae:
    def __init__(self):
        self.decode_calls = 0

    def decode(self, latents, **kwargs):
        self.decode_calls += 1
        return latents


class _FakePipe:
    def __init__(self):
        self.scheduler = _FakeScheduler()
        self.vae = _FakeVae()
        self.units = [object()]
        self.in_iteration_models = []
        self.torch_dtype = torch.float32
        self.device = torch.device("cpu")
        self.model_fn = object()
        self.step_calls = 0

    def unit_runner(self, unit, pipe, inputs, posi, nega):
        inputs["latents"] = torch.zeros(
            1, 2, inputs["height"] // 8, inputs["width"] // 8
        )
        return inputs, posi, nega

    def load_models_to_device(self, models):
        return None

    def cfg_guided_model_fn(self, model_fn, cfg, inputs, posi, nega, **kwargs):
        return torch.zeros_like(inputs["latents"])

    def step(self, scheduler, progress_id, noise_pred, **inputs):
        self.step_calls += 1
        return inputs["latents"] + 1

    def vae_output_to_image(self, output):
        return Image.new("RGB", (64, 32), (10, 20, 30))


class _FakeAtlasEngine:
    def __init__(self):
        self.calls = []

    def inference_with_atlas_latent_sync(self, *args):
        self.calls.append(args)
        return args[1]


class AtlasInferenceTests(unittest.TestCase):
    def test_single_trajectory_syncs_each_step_and_decodes_once(self):
        engine = image_inference.ImageStyleInference.__new__(
            image_inference.ImageStyleInference
        )
        engine.pipe = _FakePipe()
        atlas = Image.new("RGB", (64, 32))
        style = Image.new("RGB", (32, 32))
        synchronizer = image_inference.synchronize_hemisphere_atlas_latent
        with mock.patch.object(
            image_inference,
            "synchronize_hemisphere_atlas_latent",
            wraps=synchronizer,
        ) as synchronize:
            result = engine.inference_with_atlas_latent_sync(
                "prompt", atlas, style, 123, 3, 15.0
            )

        self.assertEqual(result.size, atlas.size)
        self.assertEqual(engine.pipe.step_calls, 3)
        self.assertEqual(synchronize.call_count, 4)
        self.assertEqual(engine.pipe.vae.decode_calls, 1)

    def test_stylize_panorama_atlas_returns_native_metadata(self):
        engine = _FakeAtlasEngine()
        content = Image.new("RGB", (64, 32), (40, 80, 120))
        style = Image.new("RGB", (16, 16))
        result, padding, atlas_size = panorama_inference.stylize_panorama(
            engine, content, style, "prompt", 123, 2, 16, 0
        )
        self.assertEqual(result.size, content.size)
        self.assertEqual(padding, 0)
        self.assertEqual(atlas_size, (64, 32))
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(engine.calls[0][1].size, atlas_size)

    def test_stylize_panorama_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "atlas.*hemisphere-latent.*legacy"):
            panorama_inference.stylize_panorama(
                _FakeAtlasEngine(), Image.new("RGB", (64, 32)),
                Image.new("RGB", (16, 16)), "prompt", 123, 2, 16, 0,
                panorama_mode="hemisphere",
            )


if __name__ == "__main__":
    unittest.main()
