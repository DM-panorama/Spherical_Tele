import math
import unittest

import torch
from torch import nn

from telestyle_spherope import (
    QwenSphericalRoPE, spherical_rope_context, spherical_width_phases,
)


class SphericalWidthTests(unittest.TestCase):
    def test_all_channels_wrap_after_one_revolution(self):
        columns = torch.tensor([-0.5, 0.0, 12.25, 127.5])
        first = spherical_width_phases(64, 128, 56, columns=columns)
        second = spherical_width_phases(64, 128, 56, columns=columns + 128)
        torch.testing.assert_close(first.cos(), second.cos(), atol=2e-5, rtol=0)
        torch.testing.assert_close(first.sin(), second.sin(), atol=2e-5, rtol=0)

    def test_polar_limit_converges_in_spherical_subspace(self):
        # The final slots have less than one cycle and always use spherical XY.
        poles = spherical_width_phases(32, 64, 56, rows=[-0.5, 31.5])
        torch.testing.assert_close(
            poles[..., -2:], poles[:, :1, -2:].expand(-1, 64, -1),
            atol=1e-6, rtol=0,
        )
        centers = spherical_width_phases(32, 64, 56)
        self.assertGreater(centers[0, :, -2:].std().item(), 0)

    def test_no_equatorial_discontinuity(self):
        phases = spherical_width_phases(32, 64, 56, rows=[15.5 - 1e-3, 15.5 + 1e-3])
        torch.testing.assert_close(phases[0], phases[1], atol=1e-5, rtol=0)

    def test_frequency_split_is_contiguous_and_cyclic_band_is_quantized(self):
        # dim=8, theta=16 gives frequencies [1, .5, .25, .125].
        # W=64: the first two pass 10%; the third fails, ending the cyclic band.
        phases = spherical_width_phases(32, 64, 8, theta=16)
        delta = phases[:, 1:] - phases[:, :-1]
        expected = torch.tensor([10, 5]) * (2 * math.pi / 64)
        torch.testing.assert_close(delta[..., :2], expected.expand(32, 63, 2), atol=2e-6, rtol=0)
        self.assertGreater(delta[..., 2].std().item(), 0.01)

    def test_rotation_preserves_norm_and_small_grids_are_finite(self):
        for height in (1, 2, 7, 64):
            phases = spherical_width_phases(height, 2 * height, 56)
            self.assertTrue(torch.isfinite(phases).all())
            rotation = torch.polar(torch.ones_like(phases), phases)
            value = torch.complex(torch.full_like(phases, 2), torch.full_like(phases, 3))
            torch.testing.assert_close((value * rotation).abs(), value.abs())

    def test_invalid_parameters(self):
        for args in ((0, 2, 56), (1, 0, 56), (1, 2, 3)):
            with self.assertRaises(ValueError):
                spherical_width_phases(*args)


class QwenRoPEIntegrationTests(unittest.TestCase):
    def setUp(self):
        from diffsynth.models.qwen_image_dit import QwenEmbedRope
        self.original = QwenEmbedRope(10000, [16, 56, 56], scale_rope=True)
        self.device = torch.device("cpu")

    def test_only_output_and_content_width_change_even_for_erp_shaped_style(self):
        shapes = [(1, 4, 8), (1, 4, 8), (1, 4, 8)]
        old_image, old_text = self.original(shapes, [5], self.device)
        saved = old_image.clone()
        image, text = QwenSphericalRoPE(self.original)(shapes, [5], self.device)
        self.assertEqual(image.dtype, old_image.dtype)
        self.assertEqual(image.device, old_image.device)
        self.assertTrue(torch.equal(image[:, :36], old_image[:, :36]))
        self.assertTrue(torch.equal(image[64:], old_image[64:]))
        self.assertTrue(torch.equal(text, old_text))
        self.assertFalse(torch.equal(image[:64, 36:], old_image[:64, 36:]))
        self.assertTrue(torch.equal(image[:32, 36:], image[32:64, 36:]))
        self.assertTrue(torch.equal(old_image, saved))
        after, _ = self.original(shapes, [5], self.device)
        self.assertTrue(torch.equal(after, saved))

    def test_context_restores_original_after_success_failure_and_size_changes(self):
        dit = nn.Module()
        dit.pos_embed = self.original
        for height in (4, 8, 4):
            shapes = [(1, height, 2 * height)] * 2 + [(1, 2, 2)]
            with spherical_rope_context(dit):
                first = dit.pos_embed(shapes, [3], self.device)[0]
                second = dit.pos_embed(shapes, [3], self.device)[0]
                self.assertTrue(torch.equal(first, second))
            self.assertIs(dit.pos_embed, self.original)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with spherical_rope_context(dit):
                raise RuntimeError("injected")
        self.assertIs(dit.pos_embed, self.original)
        with spherical_rope_context(dit, enabled=False):
            self.assertIs(dit.pos_embed, self.original)

    def test_unexpected_image_layout_is_rejected(self):
        wrapper = QwenSphericalRoPE(self.original)
        for shapes in (
            [(1, 4, 8)],
            [(1, 4, 8), (1, 2, 4), (1, 2, 2)],
            [(1, 4, 4)] * 3,
            [(2, 4, 8)] * 3,
        ):
            with self.assertRaises(ValueError):
                wrapper(shapes, [3], self.device)


if __name__ == "__main__":
    unittest.main()
