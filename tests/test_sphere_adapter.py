import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from training.checkpoint import (
    AdapterEma,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)
from training.geometry import build_sphere_geometry
from training.joint_qwen import JointQwenSphereModel
from training.sphere_adapter import SphereAdapter


class _TimeEmbedding(nn.Module):
    def forward(self, timestep, dtype):
        return torch.zeros(timestep.shape[0], 8, device=timestep.device, dtype=dtype)


class _PositionEmbedding(nn.Module):
    def forward(self, shapes, text_lengths, device):
        return None


class _NormOut(nn.Module):
    def forward(self, image, conditioning):
        return image


class _Block(nn.Module):
    def forward(self, image, text, **kwargs):
        return text, image + 0.1


class _TinyDit(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_in = nn.Linear(4, 8)
        self.txt_norm = nn.Identity()
        self.txt_in = nn.Identity()
        self.time_text_embed = _TimeEmbedding()
        self.pos_embed = _PositionEmbedding()
        self.transformer_blocks = nn.ModuleList([_Block(), _Block()])
        self.norm_out = _NormOut()
        self.proj_out = nn.Linear(8, 4)


class SphereAdapterTests(unittest.TestCase):
    def _model(self):
        adapter = SphereAdapter(
            hidden_dim=8,
            adapter_dim=8,
            num_heads=2,
            global_tokens=2,
            equator_tokens=4,
        )
        return JointQwenSphereModel(
            _TinyDit(), adapter, injection_stride=1,
            use_gradient_checkpointing=False,
        )

    def test_zero_gates_match_cross_chart_disabled(self):
        torch.manual_seed(1)
        model = self._model()
        geometry = build_sphere_geometry(2, 15, 10, 2, torch.device("cpu"))
        latent_n = torch.randn(1, 1, 4, 4)
        latent_s = torch.randn(1, 1, 4, 4)
        prompt = torch.randn(1, 3, 8)
        mask = torch.ones(1, 3, dtype=torch.long)
        arguments = (
            latent_n, latent_s, latent_n, latent_s,
            prompt, prompt, mask, mask, torch.tensor([500.0]), geometry,
        )
        enabled = model(*arguments)
        disabled = model(*arguments, disable_cross_chart=True)
        torch.testing.assert_close(enabled[0], disabled[0])
        torch.testing.assert_close(enabled[1], disabled[1])
        self.assertTrue(all(not p.requires_grad for p in model.dit.parameters()))

    def test_open_gate_propagates_adapter_gradients(self):
        torch.manual_seed(2)
        model = self._model()
        model.injection_gates.data.fill_(0.1)
        geometry = build_sphere_geometry(2, 15, 10, 2, torch.device("cpu"))
        latent = torch.randn(1, 1, 4, 4)
        prompt = torch.randn(1, 3, 8)
        mask = torch.ones(1, 3, dtype=torch.long)
        output = model(
            latent, latent, latent, latent,
            prompt, prompt, mask, mask, torch.tensor([500.0]), geometry,
        )
        (output[0].mean() + output[1].mean()).backward()
        gradients = [
            parameter.grad for parameter in model.adapter.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)

    def test_float_geometry_runs_with_bfloat16_adapter(self):
        adapter = SphereAdapter(
            hidden_dim=8, adapter_dim=8, num_heads=2,
            global_tokens=2, equator_tokens=4,
        ).to(torch.bfloat16)
        geometry = build_sphere_geometry(2, 15, 10, 2, torch.device("cpu"))
        north = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        south = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        north, south = adapter.add_position(north, south, geometry)
        updates = adapter(north, south, geometry)
        self.assertEqual(updates[0].dtype, torch.bfloat16)
        self.assertEqual(updates[1].shape, south.shape)
        self.assertTrue(torch.isfinite(updates[0]).all())

    def test_adapter_checkpoint_round_trip(self):
        model = self._model()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )
        ema = AdapterEma(model.adapter)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            save_adapter_checkpoint(
                path, model.adapter, model.injection_gates,
                ema, optimizer, 17, {"base": "test"},
            )
            model.injection_gates.data.fill_(1)
            step, metadata = load_adapter_checkpoint(
                path, model.adapter, model.injection_gates, ema, optimizer
            )
            self.assertEqual(step, 17)
            self.assertEqual(metadata["base"], "test")
            self.assertEqual(float(model.injection_gates.detach().abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
