import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from diffsynth.diffusion.base_pipeline import PipelineUnit, PipelineUnitRunner
from diffsynth.models.qwen_image_dit import QwenEmbedRope
from diffsynth.pipelines.qwen_image import QwenImageUnit_EditImageEmbedder
from telestyleimage_inference import ImageStyleInference, _prepare_edit_inputs
from telestylepanorama_inference import (
    _restore_outside_equatorial_band, main, parse_args, stylize_panorama,
)
from telestyle_spherope import QwenSphericalRoPE


class _InputUnit(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)

    def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        height, width = inputs_shared["height"] // 8, inputs_shared["width"] // 8
        inputs_shared["latents"] = torch.zeros(1, 3, height, width)
        return inputs_shared, inputs_posi, inputs_nega


class _Vae:
    def __init__(self):
        self.encoded = []
        self.decoded = []

    def encode(self, image, **kwargs):
        self.encoded.append(image.clone())
        return F.avg_pool2d(image, 8)

    def decode(self, latent, **kwargs):
        self.decoded.append(latent.clone())
        return F.interpolate(latent, scale_factor=8, mode="nearest")


class _Scheduler:
    def set_timesteps(self, steps, **kwargs):
        self.timesteps = torch.arange(steps)
        self.shift = kwargs["dynamic_shift_len"]


class _Pipe:
    def __init__(self):
        self.dit = nn.Module()
        self.dit.pos_embed = QwenEmbedRope(10000, [16, 56, 56], scale_rope=True)
        self.original = self.dit.pos_embed
        self.units = [_InputUnit(), QwenImageUnit_EditImageEmbedder()]
        self.unit_runner = PipelineUnitRunner()
        self.vae = _Vae()
        self.scheduler = _Scheduler()
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.in_iteration_models = ["dit"]
        self.model_fn = object()
        self.calls = []
        self.steps = []
        self.loads = []
        self.fail_model = False

    def load_models_to_device(self, names):
        self.loads.append(list(names))

    def preprocess_image(self, image):
        return torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255

    def cfg_guided_model_fn(self, model_fn, cfg, shared, posi, nega, **kwargs):
        shapes = [(1, x.shape[-2] // 2, x.shape[-1] // 2)
                  for x in [shared["latents"], *shared["edit_latents"]]]
        self.dit.pos_embed(shapes, [3], self.device)
        self.calls.append((shared["latents"].clone(), isinstance(self.dit.pos_embed, QwenSphericalRoPE)))
        if self.fail_model:
            raise RuntimeError("injected model failure")
        return torch.ones_like(shared["latents"])

    def step(self, scheduler, progress_id, noise_pred, **shared):
        if scheduler is not self.scheduler:
            raise AssertionError("Expected one scheduler trajectory")
        self.steps.append(progress_id)
        return shared["latents"] + noise_pred

    def vae_output_to_image(self, tensor):
        array = tensor[0].permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy()
        return Image.fromarray(array)


class SphericalInferenceTests(unittest.TestCase):
    def setUp(self):
        self.engine = ImageStyleInference.__new__(ImageStyleInference)
        self.engine.pipe = _Pipe()
        self.content = Image.fromarray(np.arange(32 * 64 * 3, dtype=np.uint8).reshape(32, 64, 3))
        self.style = Image.new("RGB", (16, 16), "red")

    def test_content_padding_crop_and_style_encoding(self):
        pipe = self.engine.pipe
        units_before = list(pipe.units)
        shared, _, _ = _prepare_edit_inputs(pipe, "prompt", self.content, self.style, 123, 2, 16)
        padded, style_tensor = pipe.vae.encoded
        content_tensor = pipe.preprocess_image(self.content)
        self.assertEqual(padded.shape[-2:], (64, 96))
        torch.testing.assert_close(padded[..., 16:48, 16:80], content_tensor)
        # North cap corresponds to reflected northern rows half a turn away.
        torch.testing.assert_close(padded[..., :16, 16:48], content_tensor[..., :16, 32:].flip(-2))
        torch.testing.assert_close(padded[..., 16:48, :16], content_tensor[..., -16:])
        torch.testing.assert_close(shared["edit_latents"][0], F.avg_pool2d(content_tensor, 8))
        torch.testing.assert_close(style_tensor, pipe.preprocess_image(self.style))
        self.assertEqual(pipe.units, units_before)
        self.assertEqual(shared["edit_image"], [self.content, self.style])
        _prepare_edit_inputs(pipe, "prompt", self.content, self.style, 123, 2)
        self.assertEqual(pipe.vae.encoded[-2].shape[-2:], (32, 64))

    def test_full_erp_uses_one_trajectory_and_restores_rope(self):
        pipe = self.engine.pipe
        with patch("telestyleimage_inference.HemisphereLatentProjector", side_effect=AssertionError("chart used")):
            result = self.engine.inference_with_spherope("prompt", self.content, self.style, 123, 3, 16)
        self.assertEqual(result.size, (64, 32))
        self.assertEqual(result.getpixel((0, 0)), (3, 3, 3))
        self.assertEqual(pipe.scheduler.shift, 8)
        self.assertEqual(pipe.steps, [0, 1, 2])
        self.assertEqual(len(pipe.calls), 3)
        for index, (latent, enabled) in enumerate(pipe.calls):
            self.assertTrue(enabled)
            torch.testing.assert_close(latent, torch.full((1, 3, 4, 8), float(index)))
        self.assertEqual(pipe.vae.decoded[0].shape[-2:], (8, 12))
        self.assertIs(pipe.dit.pos_embed, pipe.original)
        self.assertEqual(pipe.loads[-1], [])

    def test_zero_padding_and_rope_disabled_ablation(self):
        pipe = self.engine.pipe
        result = self.engine.inference_with_spherope(
            "prompt", self.content, self.style, 123, 1, 0, enable_spherope=False,
        )
        self.assertEqual(result.size, self.content.size)
        self.assertFalse(pipe.calls[0][1])
        self.assertEqual(pipe.vae.encoded[0].shape[-2:], (32, 64))
        self.assertEqual(pipe.vae.decoded[0].shape[-2:], (4, 8))

    def test_return_latents_and_model_failure_cleanup(self):
        pipe = self.engine.pipe
        latent = self.engine.inference_with_spherope(
            "prompt", self.content, self.style, 123, 1, 0, return_latents=True,
        )
        self.assertEqual(latent.shape[-2:], (4, 8))
        self.assertEqual(pipe.vae.decoded, [])
        self.assertIs(pipe.dit.pos_embed, pipe.original)
        pipe.fail_model = True
        with self.assertRaisesRegex(RuntimeError, "injected model failure"):
            self.engine.inference_with_spherope("prompt", self.content, self.style, 123, 1, 0)
        self.assertIs(pipe.dit.pos_embed, pipe.original)
        self.assertEqual(pipe.loads[-1], [])


class SphericalRoutingTests(unittest.TestCase):
    def test_default_api_aligns_and_restores_original_size(self):
        class Engine:
            def inference_with_spherope(self, prompt, content, style, seed, steps, padding_px):
                self.arguments = (content.size, style.size, seed, steps, padding_px)
                return Image.new("RGB", content.size)

        engine = Engine()
        result, padding, size = stylize_panorama(
            engine, Image.new("RGB", (100, 50)), Image.new("RGB", (16, 16)),
            "prompt", 123, 4, 256, 96, hemisphere_size=7,
        )
        self.assertEqual(result.size, (100, 50))
        self.assertEqual(size, (96, 48))
        self.assertEqual(padding, 16)
        self.assertEqual(engine.arguments, ((96, 48), (1024, 1024), 123, 4, 16))

    def test_cli_default_and_old_modes(self):
        argv = ["prog", "--content", "content.png", "--style", "style.png", "--output", "out.png"]
        for mode in (None, "spherope", "hemisphere", "legacy"):
            with patch.object(sys, "argv", argv + ([] if mode is None else ["--panorama-mode", mode])):
                self.assertEqual(parse_args().panorama_mode, mode or "spherope")

    def test_invalid_api_options_fail_before_engine_call(self):
        base = dict(engine=None, content=Image.new("RGB", (64, 32)), style=Image.new("RGB", (16, 16)),
                    prompt="prompt", seed=123, steps=4, margin_px=256, blend_px=96)
        for change in (
            {"content": Image.new("RGB", (64, 64))}, {"steps": 0},
            {"decode_padding_px": -1}, {"panorama_mode": "unknown"},
            {"enable_polar_fusion": True}, {"use_sphere_adapter": True},
            {"return_chart_images": True}, {"rgb_hard_cut_without_a1": True},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                stylize_panorama(**(base | change))

    def test_no_a1_hard_cut_routes_color_match_and_restores_after_resize(self):
        class Engine:
            def inference_with_hemisphere_rgb_hard_cut(self, *args, **kwargs):
                self.color_match_degrees = kwargs["color_match_degrees"]
                self.residual_degrees = kwargs["seam_residual_degrees"]
                self.residual_blur_degrees = kwargs[
                    "seam_residual_blur_degrees"
                ]
                self.returns_baseline = kwargs["return_hard_cut_baseline"]
                return (
                    Image.new("RGB", (64, 32), (200, 210, 220)),
                    Image.new("RGB", (64, 32), (10, 20, 30)),
                )

        engine = Engine()
        result, _, _ = stylize_panorama(
            engine, Image.new("RGB", (66, 33)), Image.new("RGB", (16, 16)),
            "prompt", 123, 1, 256, 96, panorama_mode="hemisphere",
            rgb_hard_cut_without_a1=True, hemisphere_color_match_degrees=6.0,
        )
        array = np.asarray(result)
        latitude = 90.0 - (np.arange(33) + 0.5) * (180.0 / 33)
        self.assertEqual(engine.color_match_degrees, 6.0)
        self.assertEqual(engine.residual_degrees, 2.0)
        self.assertEqual(engine.residual_blur_degrees, 0.5)
        self.assertTrue(engine.returns_baseline)
        self.assertTrue((array[np.abs(latitude) >= 6.0] == (10, 20, 30)).all())

    def test_equatorial_band_restore_keeps_outside_rgb_exact(self):
        baseline = Image.new("RGB", (20, 10), (10, 20, 30))
        repaired = Image.new("RGB", (20, 10), (200, 210, 220))
        result = np.asarray(
            _restore_outside_equatorial_band(repaired, baseline, 20.0)
        )
        latitude = 90.0 - (np.arange(10) + 0.5) * 18.0
        inside = np.abs(latitude) < 20.0
        self.assertTrue((result[~inside] == (10, 20, 30)).all())
        self.assertTrue((result[inside] == (200, 210, 220)).all())

    def test_color_match_band_cannot_exceed_overlap(self):
        with self.assertRaisesRegex(ValueError, "color-match-degrees"):
            stylize_panorama(
                None, Image.new("RGB", (64, 32)), Image.new("RGB", (16, 16)),
                "prompt", 123, 4, 256, 96, panorama_mode="hemisphere",
                rgb_hard_cut_without_a1=True,
                hemisphere_overlap_degrees=15.0,
                hemisphere_color_match_degrees=16.0,
            )

    def test_residual_parameters_are_validated_before_engine_call(self):
        base = dict(
            engine=None, content=Image.new("RGB", (64, 32)),
            style=Image.new("RGB", (16, 16)), prompt="prompt", seed=123,
            steps=4, margin_px=256, blend_px=96, panorama_mode="hemisphere",
            rgb_hard_cut_without_a1=True,
        )
        with self.assertRaisesRegex(ValueError, "seam-residual-degrees"):
            stylize_panorama(
                **base, hemisphere_seam_residual_degrees=16.0
            )
        with self.assertRaisesRegex(ValueError, "residual-blur-degrees"):
            stylize_panorama(
                **base, hemisphere_seam_residual_blur_degrees=0.0
            )

    def test_cli_rejects_incompatible_options_before_loading_models(self):
        base = ["prog", "--content", "unused", "--style", "unused", "--output", "unused"]
        for options in (["--a1-checkpoint", "unused"], ["--enable-polar-fusion"], ["--save-a1-chart-images"]):
            with patch.object(sys, "argv", base + options), patch("telestylepanorama_inference.ImageStyleInference") as engine:
                with self.assertRaises(ValueError):
                    main()
                engine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
