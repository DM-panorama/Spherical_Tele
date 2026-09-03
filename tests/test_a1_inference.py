import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

from telestylepanorama_inference import (
    _a1_ab_output_paths,
    _a1_chart_output_paths,
    _rgb_comparison_metrics,
    _validate_a1_args,
    stylize_panorama,
)
from training.checkpoint import sha256_file
from training.joint_qwen import JointQwenSphereModel
from training.sphere_adapter import SphereAdapter
from telestyleimage_inference import ImageStyleInference


class _PositionEmbedding:
    def __init__(self):
        self.shapes = None

    def __call__(self, shapes, text_lengths, device):
        self.shapes = shapes
        return torch.empty(0), torch.empty(0)


class _FakeDit(nn.Module):
    def __init__(self, blocks=0):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.transformer_blocks = nn.ModuleList(
            [nn.Identity() for _ in range(blocks)]
        )
        self.img_in = nn.Identity()
        self.txt_norm = nn.Identity()
        self.txt_in = nn.Identity()
        self.pos_embed = _PositionEmbedding()

    def time_text_embed(self, timestep, dtype):
        return torch.zeros(1, 4, dtype=dtype)


class _FakeAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))


class A1MultiEditTests(unittest.TestCase):
    def test_prepare_branch_accepts_multiple_edit_latents(self):
        dit = _FakeDit()
        model = JointQwenSphereModel(dit, _FakeAdapter(), 1, False)
        latents = torch.zeros(1, 1, 4, 4)
        prompt = torch.zeros(1, 2, 4)
        mask = torch.ones(1, 2, dtype=torch.bool)
        timestep = torch.ones(1)

        state = model._prepare_branch(
            latents,
            [torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 2, 2)],
            prompt,
            mask,
            timestep,
        )

        self.assertEqual(state.output_tokens, 4)
        self.assertEqual(state.image.shape[1], 9)
        self.assertEqual(
            dit.pos_embed.shapes,
            [(1, 2, 2), (1, 2, 2), (1, 1, 1)],
        )

    def test_prepare_branch_preserves_single_edit_interface(self):
        dit = _FakeDit()
        model = JointQwenSphereModel(dit, _FakeAdapter(), 1, False)
        state = model._prepare_branch(
            torch.zeros(1, 1, 4, 4),
            torch.zeros(1, 1, 4, 4),
            torch.zeros(1, 2, 4),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1),
        )
        self.assertEqual(state.image.shape[1], 8)


class A1CheckpointLoaderTests(unittest.TestCase):
    def test_loader_uses_ema_adapter_and_raw_injection_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transformer = root / "transformer"
            transformer.mkdir()
            index = transformer / "diffusion_pytorch_model.safetensors.index.json"
            index.write_text("{}", encoding="utf-8")
            config = root / "a1.yaml"
            config.write_text(
                "stage: A1\n"
                "model:\n"
                "  hidden_dim: 4\n"
                "  adapter_dim: 4\n"
                "  num_heads: 1\n"
                "  global_tokens: 2\n"
                "  equator_tokens: 2\n"
                "  injection_stride: 1\n",
                encoding="utf-8",
            )
            adapter = SphereAdapter(4, 4, 1, 2, 2)
            ema_state = {
                name: value.detach().clone()
                for name, value in adapter.state_dict().items()
            }
            ema_state["position_gate"] = torch.tensor(0.25)
            checkpoint = root / "adapter.pt"
            torch.save(
                {
                    "format": "telestyle-sphere-adapter-a1-v1",
                    "step": 75,
                    "adapter": adapter.state_dict(),
                    "injection_gates": torch.tensor([0.5]),
                    "ema": {"decay": 0.99, "shadow": ema_state},
                    "metadata": {
                        "chart_size": 32,
                        "base_transformer_index_sha256": sha256_file(index),
                    },
                },
                checkpoint,
            )
            engine = ImageStyleInference.__new__(ImageStyleInference)
            engine.device = torch.device("cpu")
            engine.model_dir = str(root)
            engine.pipe = type("Pipe", (), {
                "dit": _FakeDit(blocks=1),
                "torch_dtype": torch.float32,
            })()
            engine.sphere_model = None
            engine.sphere_config = None
            engine.sphere_checkpoint_info = None

            info = engine.load_sphere_adapter(config, checkpoint, 32)

            self.assertEqual(info["step"], 75)
            self.assertAlmostEqual(
                float(engine.sphere_model.adapter.position_gate.detach()), 0.25
            )
            self.assertAlmostEqual(
                float(engine.sphere_model.injection_gates[0].detach()), 0.5
            )

    def test_loader_rejects_chart_size_mismatch(self):
        engine = ImageStyleInference.__new__(ImageStyleInference)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "adapter.pt"
            torch.save(
                {
                    "format": "telestyle-sphere-adapter-a1-v1",
                    "metadata": {"chart_size": 32},
                },
                checkpoint,
            )
            config = Path(directory) / "a1.yaml"
            config.write_text("stage: A1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "chart size"):
                engine.load_sphere_adapter(config, checkpoint, 64)


class _FakePanoramaEngine:
    def __init__(self):
        self.baseline_calls = 0
        self.a1_calls = 0
        self.no_a1_calls = 0

    def inference_with_hemisphere_latent_sync(self, *args):
        self.baseline_calls += 1
        return Image.new("RGB", (args[7], args[6]), "black")

    def inference_with_hemisphere_rgb_hard_cut(self, *args, **kwargs):
        self.no_a1_calls += 1
        return Image.new("RGB", (args[7], args[6]), "green")

    def inference_with_hemisphere_sphere_adapter(self, *args, **kwargs):
        self.a1_calls += 1
        result = Image.new("RGB", (args[7], args[6]), "white")
        if kwargs.get("return_chart_images"):
            chart_size = args[1].width
            return (
                result, Image.new("RGB", (chart_size, chart_size), "red"),
                Image.new("RGB", (chart_size, chart_size), "blue"),
            )
        return result


class _FakeScheduler:
    def set_timesteps(self, steps, **kwargs):
        self.timesteps = [torch.tensor(index) for index in range(steps)]


class _FakeVae:
    def __init__(self):
        self.decode_calls = 0

    def decode(self, latent, **kwargs):
        self.decode_calls += 1
        return torch.zeros(
            latent.shape[0], 3, latent.shape[-2] * 8, latent.shape[-1] * 8
        )


class _FakeA1Pipe:
    def __init__(self):
        self.scheduler = _FakeScheduler()
        self.vae = _FakeVae()
        self.units = [object()]
        self.in_iteration_models = []
        self.torch_dtype = torch.float32
        self.device = torch.device("cpu")
        self.model_fn = object()
        self.model_calls = 0

    def unit_runner(self, unit, pipe, inputs, posi, nega):
        latent = torch.zeros(
            1, 2, inputs["height"] // 8, inputs["width"] // 8
        )
        inputs["latents"] = latent
        inputs["noise"] = latent.clone()
        inputs["edit_latents"] = [latent.clone(), latent.clone()]
        posi["prompt_emb"] = torch.zeros(1, 2, 4)
        posi["prompt_emb_mask"] = torch.ones(1, 2, dtype=torch.bool)
        return inputs, posi, nega

    def load_models_to_device(self, models):
        return None

    def cfg_guided_model_fn(
        self, model_fn, cfg, inputs, posi, nega, **kwargs
    ):
        self.model_calls += 1
        return torch.zeros_like(inputs["latents"])

    def step(self, scheduler, progress_id, noise_pred, **inputs):
        return inputs["latents"]

    def vae_output_to_image(self, tensor):
        return Image.new("RGB", (tensor.shape[-1], tensor.shape[-2]))


class _FakeSphereModel:
    def __call__(self, north, south, *args, **kwargs):
        return torch.zeros_like(north), torch.zeros_like(south)


class A1RgbHardCutTests(unittest.TestCase):
    def test_no_a1_path_decodes_independent_charts_and_hard_cuts_rgb(self):
        engine = ImageStyleInference.__new__(ImageStyleInference)
        engine.pipe = _FakeA1Pipe()
        chart = Image.new("RGB", (32, 32))

        result = engine.inference_with_hemisphere_rgb_hard_cut(
            "prompt", chart, chart, chart, 123, 2, 16, 32, 15.0, 10.0
        )

        self.assertEqual(engine.pipe.model_calls, 4)
        self.assertEqual(engine.pipe.vae.decode_calls, 2)
        self.assertEqual(result.size, (32, 16))

    def test_a1_decodes_two_charts_without_decoding_fused_latent(self):
        engine = ImageStyleInference.__new__(ImageStyleInference)
        engine.pipe = _FakeA1Pipe()
        engine.sphere_model = _FakeSphereModel()
        engine.sphere_config = SimpleNamespace(
            data=SimpleNamespace(consistency_degrees=10.0),
            model=SimpleNamespace(correspondence_neighbors=1),
        )
        engine.sphere_checkpoint_info = {"metadata": {"chart_size": 32}}
        chart = Image.new("RGB", (32, 32))

        result, north, south = engine.inference_with_hemisphere_sphere_adapter(
            "prompt", chart, chart, chart, 123, 2, 16, 32,
            15.0, 0, return_chart_images=True,
        )

        self.assertEqual(engine.pipe.vae.decode_calls, 2)
        self.assertEqual(result.size, (32, 16))
        self.assertEqual(north.size, (32, 32))
        self.assertEqual(south.size, (32, 32))


class A1PanoramaRoutingTests(unittest.TestCase):
    def test_latent_and_chart_return_modes_are_mutually_exclusive(self):
        engine = ImageStyleInference.__new__(ImageStyleInference)
        chart = Image.new("RGB", (32, 32))
        with self.assertRaisesRegex(ValueError, "cannot both be enabled"):
            engine.inference_with_hemisphere_sphere_adapter(
                "prompt", chart, chart, chart, 123, 1, 16, 32,
                return_latents=True, return_chart_images=True,
            )

    def test_default_and_a1_paths_route_independently(self):
        content = Image.new("RGB", (64, 32), "gray")
        style = Image.new("RGB", (16, 16), "blue")
        engine = _FakePanoramaEngine()
        baseline, _, _ = stylize_panorama(
            engine, content, style, "prompt", 123, 4, 16, 0,
            hemisphere_size=32, use_sphere_adapter=False,
        )
        adapter, _, _ = stylize_panorama(
            engine, content, style, "prompt", 123, 4, 16, 0,
            hemisphere_size=32, use_sphere_adapter=True,
        )
        self.assertEqual(engine.baseline_calls, 1)
        no_a1, _, _ = stylize_panorama(
            engine, content, style, "prompt", 123, 4, 16, 0,
            hemisphere_size=32, rgb_hard_cut_without_a1=True,
        )
        self.assertEqual(engine.a1_calls, 1)
        self.assertEqual(engine.no_a1_calls, 1)
        self.assertEqual(no_a1.size, content.size)
        self.assertEqual(baseline.size, content.size)
        self.assertEqual(adapter.size, content.size)

        result = stylize_panorama(
            engine, content, style, "prompt", 123, 4, 16, 0,
            hemisphere_size=32, use_sphere_adapter=True,
            return_chart_images=True,
        )
        self.assertEqual(result[0].size, content.size)
        self.assertEqual(result[3].size, (32, 32))
        self.assertEqual(result[4].size, (32, 32))


class A1ComparisonHelpersTests(unittest.TestCase):
    def test_output_paths_are_derived_from_a1_output(self):
        paths = _a1_ab_output_paths(Path("outputs/test.png"))
        self.assertEqual(paths[0], Path("outputs/test_baseline.png"))
        self.assertEqual(
            paths[1], Path("outputs/test_no_a1_rgb_hardcut.png")
        )
        self.assertEqual(paths[2], Path("outputs/test_report.json"))
        named = _a1_ab_output_paths(
            Path("outputs/telestyle_a1_rgb_hardcut.png")
        )
        self.assertEqual(
            named[1], Path("outputs/telestyle_no_a1_rgb_hardcut.png")
        )
        charts = _a1_chart_output_paths(Path("outputs/test.png"))
        self.assertEqual(charts[0], Path("outputs/test_north_chart.png"))
        self.assertEqual(charts[1], Path("outputs/test_south_chart.png"))

    def test_comparison_sheet_and_metrics(self):
        baseline = Image.new("RGB", (8, 4), "black")
        adapter = Image.new("RGB", (8, 4), "white")
        metrics = _rgb_comparison_metrics(baseline, adapter)
        self.assertAlmostEqual(metrics["pixel_mae"], 1.0)
        self.assertAlmostEqual(metrics["pixel_max_difference"], 1.0)
        self.assertAlmostEqual(metrics["baseline_left_right_seam_l1"], 0.0)
        self.assertAlmostEqual(metrics["a1_left_right_seam_l1"], 0.0)

    def test_ab_mode_requires_checkpoint(self):
        args = argparse.Namespace(
            a1_ab_test=True, a1_checkpoint=None, panorama_mode="hemisphere"
        )
        with self.assertRaisesRegex(ValueError, "requires --a1-checkpoint"):
            _validate_a1_args(args)

    def test_chart_output_requires_checkpoint(self):
        args = argparse.Namespace(
            a1_ab_test=False, a1_checkpoint=None, panorama_mode="hemisphere",
            save_a1_chart_images=True,
        )
        with self.assertRaisesRegex(ValueError, "requires --a1-checkpoint"):
            _validate_a1_args(args)

    def test_a1_rejects_legacy_mode(self):
        args = argparse.Namespace(
            a1_ab_test=False, a1_checkpoint="adapter.pt", panorama_mode="legacy"
        )
        with self.assertRaisesRegex(ValueError, "hemisphere"):
            _validate_a1_args(args)


if __name__ == "__main__":
    unittest.main()
