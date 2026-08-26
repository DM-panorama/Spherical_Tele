import unittest

import numpy as np
import torch
from PIL import Image

from telestyle_spherical import (
    HemisphereLatentProjector,
    extract_stereographic_hemisphere,
    make_spherical_latent_canvas,
    SphericalLatentProjector,
    early_polar_guidance_strength,
    latitude_adaptive_circular_lowpass,
    limit_polar_latent_detail,
    make_polar_latitude_weight,
    make_circular_latent_canvas,
    make_rotated_latent_canvas,
)


class SphericalReprojectionTests(unittest.TestCase):
    def test_zero_degree_round_trip_is_identity(self):
        latents = torch.randn(1, 4, 16, 32)
        projector = SphericalLatentProjector(16, 32, 0.0, 45.0, 75.0, torch.device("cpu"))
        restored = projector.b_to_a(projector.a_to_b(latents))
        self.assertTrue(torch.allclose(restored, latents, atol=5e-5, rtol=5e-5))

    def test_ninety_degree_rotation_round_trip_preserves_smooth_signal(self):
        y = torch.linspace(-1.0, 1.0, 32).view(1, 1, 32, 1)
        x = torch.linspace(-1.0, 1.0, 64).view(1, 1, 1, 64)
        latents = torch.cat((x.expand(1, 1, 32, 64), y.expand(1, 1, 32, 64)), dim=1)
        projector = SphericalLatentProjector(32, 64, 90.0, 45.0, 75.0, torch.device("cpu"))
        restored = projector.b_to_a(projector.a_to_b(latents))
        self.assertLess((restored - latents).abs().mean().item(), 0.04)

    def test_polar_weight_is_zero_at_equator_and_one_at_poles(self):
        projector = SphericalLatentProjector(180, 32, 90.0, 45.0, 75.0, torch.device("cpu"))
        weight = projector.polar_weight[0, 0, :, 0]
        self.assertEqual(weight[90].item(), 0.0)
        self.assertEqual(weight[0].item(), 1.0)
        self.assertEqual(weight[-1].item(), 1.0)
        self.assertGreater(weight[30].item(), 0.0)
        self.assertLess(weight[30].item(), 1.0)

    def test_x_axis_rotation_moves_a_pole_to_equatorial_source_latitude(self):
        height, width = 32, 64
        projector = SphericalLatentProjector(height, width, 90.0, 45.0, 75.0, torch.device("cpu"))
        # The north-pole output pixel in B samples an equatorial source point in A.
        source_y = (projector.a_to_b_grid[0, width // 2, 1] + 1) * height / 2 - 0.5
        self.assertLess(abs(source_y.item() - (height - 1) / 2), 0.75)

    def test_circular_longitude_sampling_has_no_zero_seam(self):
        latents = torch.ones(1, 1, 16, 32)
        projector = SphericalLatentProjector(16, 32, 27.0, 45.0, 75.0, torch.device("cpu"))
        rotated = projector.a_to_b(latents)
        self.assertGreater(rotated.min().item(), 0.999)

    def test_fusion_strength_limits_b_contribution(self):
        projector = SphericalLatentProjector(16, 32, 90.0, 20.0, 30.0, torch.device("cpu"))
        a = torch.zeros(1, 1, 16, 32)
        b = torch.ones_like(a)
        fused = projector.fuse_a_with_b(a, b, strength=0.7)
        self.assertAlmostEqual(fused[0, 0, 0, 0].item(), 0.7, places=5)
        self.assertEqual(fused[0, 0, 8, 0].item(), 0.0)

    def test_polar_detail_limiter_preserves_equator_and_means_full_pole_row(self):
        source = torch.zeros(1, 1, 180, 32)
        source[0, 0, 0, 0] = 1.0
        source[0, 0, 90, 3] = 1.0
        weight = make_polar_latitude_weight(180, 32, 65.0, 88.0, torch.device("cpu"))
        limited = limit_polar_latent_detail(source, weight, max_radius=8, pole_mean_rows=1)
        self.assertTrue(torch.equal(limited[:, :, 90], source[:, :, 90]))
        self.assertTrue(torch.allclose(limited[:, :, 0], limited[:, :, 0, :1].expand_as(limited[:, :, 0])))
        self.assertGreater(limited[0, 0, 0, -1].item(), 0.0)

    def test_polar_detail_limiter_does_not_mean_rows_before_full_weight(self):
        source = torch.randn(1, 1, 16, 32)
        weight = make_polar_latitude_weight(16, 32, 65.0, 88.0, torch.device("cpu"))
        limited = limit_polar_latent_detail(source, weight, max_radius=0, pole_mean_rows=1)
        self.assertTrue(torch.equal(limited, source))
        with self.assertRaises(ValueError):
            limit_polar_latent_detail(source, weight, max_radius=4, pole_mean_rows=-1)

    def test_latitude_adaptive_lowpass_preserves_equator_and_blurs_poles(self):
        source = torch.zeros(1, 1, 5, 16)
        source[0, 0, 0, 0] = 1.0
        source[0, 0, 2, 3] = 1.0
        weight = torch.zeros(1, 1, 5, 16)
        weight[:, :, 0] = 1.0
        filtered = latitude_adaptive_circular_lowpass(source, weight, max_radius=4)
        self.assertTrue(torch.equal(filtered[:, :, 2], source[:, :, 2]))
        self.assertLess(filtered[0, 0, 0, 0].item(), 1.0)
        self.assertGreater(filtered[0, 0, 0, -1].item(), 0.0)

    def test_latitude_adaptive_lowpass_handles_zero_and_narrow_width(self):
        source = torch.randn(1, 2, 4, 3)
        weight = torch.ones(1, 1, 4, 3)
        self.assertTrue(torch.equal(latitude_adaptive_circular_lowpass(source, weight, 0), source))
        filtered = latitude_adaptive_circular_lowpass(source, weight, 8)
        self.assertEqual(filtered.shape, source.shape)
        with self.assertRaises(ValueError):
            latitude_adaptive_circular_lowpass(source, weight, -1)

    def test_early_guidance_schedule_uses_only_initial_steps(self):
        strengths = [early_polar_guidance_strength(step, 2) for step in range(4)]
        self.assertEqual(strengths, [0.35, 0.2, 0.0, 0.0])
        self.assertAlmostEqual(early_polar_guidance_strength(1, 3, 0.5), 0.1)
        self.assertAlmostEqual(early_polar_guidance_strength(2, 3), 0.12)
        with self.assertRaises(ValueError):
            early_polar_guidance_strength(0, 0)

    def test_rotated_initial_noise_keeps_wrapped_canvas_order(self):
        projector = SphericalLatentProjector(16, 32, 90.0, 45.0, 75.0, torch.device("cpu"))
        latents_a = torch.arange(16 * 40, dtype=torch.float32).view(1, 1, 16, 40)
        original = latents_a.clone()
        latents_b = make_rotated_latent_canvas(latents_a, projector, centre_x=4, centre_width=32)
        expected_centre = projector.a_to_b(latents_a[..., 4:36])
        self.assertEqual(latents_b.shape, latents_a.shape)
        self.assertTrue(torch.equal(latents_b[..., 4:36], expected_centre))
        self.assertTrue(torch.equal(latents_b[..., :4], expected_centre[..., -4:]))
        self.assertTrue(torch.equal(latents_b[..., 36:], expected_centre[..., :4]))
        self.assertTrue(torch.equal(latents_a, original))

    def test_aligned_b_prediction_keeps_wrapped_canvas_order(self):
        projector = SphericalLatentProjector(16, 32, 90.0, 45.0, 75.0, torch.device("cpu"))
        b_prediction = torch.arange(16 * 32, dtype=torch.float32).view(1, 1, 16, 32)
        aligned = projector.b_to_a(b_prediction)
        canvas = make_circular_latent_canvas(aligned, left_width=4, right_width=6)
        self.assertEqual(canvas.shape, (1, 1, 16, 42))
        self.assertTrue(torch.equal(canvas[..., :4], aligned[..., -4:]))
        self.assertTrue(torch.equal(canvas[..., 4:36], aligned))
        self.assertTrue(torch.equal(canvas[..., 36:], aligned[..., :6]))

    def test_temporary_b_canvas_is_built_from_a_without_mutating_a(self):
        centre = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8)
        original = centre.clone()
        canvas = make_circular_latent_canvas(centre, left_width=2, right_width=3)
        self.assertTrue(torch.equal(centre, original))
        self.assertTrue(torch.equal(canvas[0, 0, 0], torch.tensor([6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2])))

    def test_stereographic_hemisphere_has_valid_square_content(self):
        image = Image.new("RGB", (64, 32), (40, 80, 120))
        north = extract_stereographic_hemisphere(image, 32, 15.0, north=True)
        south = extract_stereographic_hemisphere(image, 32, 15.0, north=False)
        self.assertEqual(north.size, (32, 32))
        self.assertEqual(south.size, (32, 32))
        expected = torch.tensor([40, 80, 120], dtype=torch.uint8)
        self.assertTrue(torch.equal(torch.from_numpy(np.asarray(north).copy()[0, 0]), expected))
        self.assertTrue(torch.equal(torch.from_numpy(np.asarray(south).copy()[-1, -1]), expected))

    def test_hemisphere_sync_preserves_non_overlap_and_inputs(self):
        projector = HemisphereLatentProjector(32, 15.0, torch.device("cpu"))
        north = torch.zeros(1, 1, 32, 32)
        south = torch.ones_like(north)
        north_original = north.clone()
        south_original = south.clone()
        synced_north, synced_south = projector.synchronize(north, south)
        self.assertTrue(torch.equal(north, north_original))
        self.assertTrue(torch.equal(south, south_original))
        self.assertTrue(torch.equal(
            synced_north.masked_select(~projector.north_overlap),
            north.masked_select(~projector.north_overlap),
        ))
        self.assertTrue(torch.equal(
            synced_south.masked_select(~projector.south_overlap),
            south.masked_select(~projector.south_overlap),
        ))
        self.assertGreater(
            synced_north.masked_select(projector.north_overlap).max().item(), 0.0
        )
        self.assertLess(
            synced_south.masked_select(projector.south_overlap).min().item(), 1.0
        )

    def test_hemisphere_composition_preserves_uniform_latents(self):
        projector = HemisphereLatentProjector(16, 15.0, torch.device("cpu"))
        north = torch.full((1, 2, 16, 16), 3.0)
        south = torch.full_like(north, 3.0)
        synced_north, synced_south = projector.synchronize(north, south)
        erp = projector.compose_erp(synced_north, synced_south, 8, 16)
        self.assertEqual(erp.shape, (1, 2, 8, 16))
        self.assertTrue(torch.allclose(erp, torch.full_like(erp, 3.0), atol=1e-6))

    def test_spherical_latent_canvas_uses_cross_pole_half_turn(self):
        centre = torch.arange(3 * 8, dtype=torch.float32).view(1, 1, 3, 8)
        canvas = make_spherical_latent_canvas(
            centre, horizontal_padding=2, vertical_padding=1
        )
        self.assertEqual(canvas.shape, (1, 1, 5, 12))
        self.assertTrue(torch.equal(canvas[..., 1:4, 2:10], centre))
        expected_north = torch.roll(centre[..., 0, :], shifts=4, dims=-1)
        expected_south = torch.roll(centre[..., -1, :], shifts=4, dims=-1)
        self.assertTrue(torch.equal(canvas[..., 0, 2:10], expected_north))
        self.assertTrue(torch.equal(canvas[..., -1, 2:10], expected_south))
        self.assertTrue(torch.equal(canvas[..., :2], canvas[..., 8:10]))
        self.assertTrue(torch.equal(canvas[..., -2:], canvas[..., 2:4]))


    def test_hemisphere_round_trip_preserves_smooth_latitude_orientation(self):
        height, width = 64, 128
        latitude_ramp = np.linspace(255, 0, height, dtype=np.uint8)[:, None]
        channel = np.repeat(latitude_ramp, width, axis=1)
        image = Image.fromarray(np.stack((channel, channel, channel), axis=-1), "RGB")
        north = extract_stereographic_hemisphere(image, 64, 15.0, north=True)
        south = extract_stereographic_hemisphere(image, 64, 15.0, north=False)
        north_tensor = torch.from_numpy(np.asarray(north).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        south_tensor = torch.from_numpy(np.asarray(south).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        reference = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        projector = HemisphereLatentProjector(64, 15.0, torch.device("cpu"))
        restored = projector.compose_erp(
            north_tensor, south_tensor, height, width
        )
        self.assertLess((restored - reference).abs().mean().item(), 0.002)
        self.assertLess((restored - reference).abs().max().item(), 0.02)


if __name__ == "__main__":
    unittest.main()
