import unittest
from unittest import mock
from pathlib import Path

import torch
from PIL import Image

import telestyleimage_inference as image_inference
from telestylepanorama_inference import (
    _hemisphere_chart_output_paths,
    stylize_panorama,
)


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
        self.seeds = []
        self.model_calls = 0
        self.step_calls = 0

    def unit_runner(self, unit, pipe, inputs, posi, nega):
        self.seeds.append(inputs["seed"])
        inputs["latents"] = torch.zeros(
            1, 2, inputs["height"] // 8, inputs["width"] // 8
        )
        return inputs, posi, nega

    def load_models_to_device(self, models):
        return None

    def cfg_guided_model_fn(self, model_fn, cfg, inputs, posi, nega, **kwargs):
        self.model_calls += 1
        return torch.zeros_like(inputs["latents"])

    def step(self, scheduler, progress_id, noise_pred, **inputs):
        self.step_calls += 1
        return inputs["latents"]

    def vae_output_to_image(self, output):
        return Image.new(
            "RGB", (output.shape[-1] * 8, output.shape[-2] * 8)
        )


class HemisphereRgbInferenceTests(unittest.TestCase):
    def test_rgb_erp_uses_only_two_chart_decodes(self):
        engine = image_inference.ImageStyleInference.__new__(
            image_inference.ImageStyleInference
        )
        engine.pipe = _FakePipe()
        chart = Image.new("RGB", (32, 32))
        coupling = (
            image_inference.HemisphereLatentProjector
            .couple_with_shared_equator
        )
        with mock.patch.object(
            image_inference.HemisphereLatentProjector,
            "couple_with_shared_equator", autospec=True,
            side_effect=coupling,
        ) as couple:
            result, north, south = (
                engine.inference_with_hemisphere_latent_sync(
                    "prompt", chart, chart, chart, 123, 2, 16, 32,
                    15.0, 0, return_chart_images=True,
                )
            )

        self.assertEqual(couple.call_count, 3)
        self.assertEqual(engine.pipe.seeds, [123, 123])
        self.assertEqual(engine.pipe.model_calls, 4)
        self.assertEqual(engine.pipe.step_calls, 4)
        self.assertEqual(engine.pipe.vae.decode_calls, 2)
        self.assertEqual(result.size, (32, 16))
        self.assertEqual(north.size, (32, 32))
        self.assertEqual(south.size, (32, 32))

    def test_latent_and_chart_return_modes_are_mutually_exclusive(self):
        engine = image_inference.ImageStyleInference.__new__(
            image_inference.ImageStyleInference
        )
        engine.pipe = _FakePipe()
        chart = Image.new("RGB", (32, 32))
        with self.assertRaisesRegex(ValueError, "cannot both be enabled"):
            engine.inference_with_hemisphere_latent_sync(
                "prompt", chart, chart, chart, 123, 1, 16, 32,
                return_latents=True, return_chart_images=True,
            )

    def test_rgb_blend_cannot_exceed_latent_overlap(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            stylize_panorama(
                object(), Image.new("RGB", (64, 32)),
                Image.new("RGB", (16, 16)), "prompt", 123, 1, 16, 0,
                hemisphere_overlap_degrees=15.0,
                hemisphere_rgb_blend_degrees=16.0,
            )

    def test_chart_output_paths_follow_erp_name(self):
        north, south = _hemisphere_chart_output_paths(
            Path("outputs/panorama.png")
        )
        self.assertEqual(north, Path("outputs/panorama_north_chart.png"))
        self.assertEqual(south, Path("outputs/panorama_south_chart.png"))


if __name__ == "__main__":
    unittest.main()
