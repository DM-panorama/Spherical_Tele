import unittest

import numpy as np
import torch
from PIL import Image

from telestyle_spherical import extract_stereographic_hemisphere

from training.geometry import (
    build_sphere_geometry,
    correct_equatorial_seam_residual,
    match_equatorial_low_frequency,
    reproject_native_charts,
    swap_sphere_geometry,
    rotate_erp,
)
from training.losses import geometric_losses


class A1GeometryLossTests(unittest.TestCase):
    def test_identity_rotation_preserves_pixel_centres(self):
        source = torch.rand(1, 3, 16, 32)
        result = rotate_erp(source, 0.0, 0.0, 0.0)
        torch.testing.assert_close(result, source, atol=5e-6, rtol=0)

    def test_correspondence_indices_are_valid_and_in_overlap(self):
        geometry = build_sphere_geometry(
            token_size=8,
            overlap_degrees=15,
            consistency_degrees=10,
            neighbors=4,
            device=torch.device("cpu"),
        )
        self.assertEqual(geometry.north_to_south_indices.shape, (64, 4))
        self.assertTrue((geometry.north_to_south_indices >= 0).all())
        self.assertTrue((geometry.north_to_south_indices < 64).all())

        swapped = swap_sphere_geometry(geometry)
        torch.testing.assert_close(
            swapped.north_directions, geometry.south_directions
        )
        torch.testing.assert_close(
            swapped.north_to_south_indices,
            geometry.south_to_north_indices,
        )
        self.assertTrue(geometry.north_overlap.any())
        self.assertTrue(geometry.south_overlap.any())

    def test_identical_constant_charts_have_no_hard_seam(self):
        north = torch.full((1, 3, 32, 32), 0.25)
        south = north.clone()
        projected = reproject_native_charts(
            north, south, 15, 10, 16, 32, antialias_scale=2
        )
        losses = geometric_losses(
            projected.north,
            projected.south,
            projected.hard_cut,
            projected.consistency_mask,
            projected.latitude_degrees,
        )
        self.assertLess(float(losses["rgb"]), 0.0011)
        self.assertEqual(float(losses["gradient"]), 0.0)
        self.assertEqual(float(losses["laplacian"]), 0.0)
        self.assertEqual(float(losses["seam"]), 0.0)
        self.assertEqual(projected.hard_cut.shape, (1, 3, 16, 32))

    def test_extracted_charts_reproject_without_a_south_half_turn(self):
        height, width = 32, 64
        longitude = np.arange(width, dtype=np.float32)
        row = np.stack(
            (
                127.5 + 127.5 * np.sin(2 * np.pi * longitude / width),
                127.5 + 127.5 * np.cos(2 * np.pi * longitude / width),
                longitude / (width - 1) * 255,
            ),
            axis=-1,
        )
        array = np.broadcast_to(row[None], (height, width, 3)).astype(np.uint8)
        source_image = Image.fromarray(array, "RGB")

        def as_tensor(image):
            return (
                torch.from_numpy(np.asarray(image).copy())
                .permute(2, 0, 1).unsqueeze(0).float() / 255
            )

        north = as_tensor(
            extract_stereographic_hemisphere(source_image, 64, 15, north=True)
        )
        south = as_tensor(
            extract_stereographic_hemisphere(source_image, 64, 15, north=False)
        )
        source = as_tensor(source_image)
        aligned = reproject_native_charts(
            north, south, 15, 10, height, width, antialias_scale=1,
            south_yaw_degrees=0.0,
        )
        shifted = reproject_native_charts(
            north, south, 15, 10, height, width, antialias_scale=1,
            south_yaw_degrees=180.0,
        )
        aligned_error = (aligned.hard_cut - source).abs().mean()
        shifted_error = (shifted.hard_cut - source).abs().mean()
        self.assertLess(float(aligned_error), 0.01)
        self.assertGreater(float(shifted_error), 0.25)

    def test_south_yaw_half_turn_only_rotates_south_erp(self):
        axis = torch.linspace(0.0, 1.0, 32)
        longitude_chart = axis.view(1, 1, 1, 32).expand(1, 1, 32, 32)
        unrotated = reproject_native_charts(
            longitude_chart, longitude_chart, 15, 10, 16, 32,
            antialias_scale=1,
        )
        rotated = reproject_native_charts(
            longitude_chart, longitude_chart, 15, 10, 16, 32,
            antialias_scale=1, south_yaw_degrees=180.0,
        )

        torch.testing.assert_close(rotated.north, unrotated.north)
        south_rows = slice(8, None)
        expected = torch.roll(unrotated.south, shifts=16, dims=-1)
        torch.testing.assert_close(
            rotated.south[..., south_rows, :],
            expected[..., south_rows, :],
            atol=2e-5,
            rtol=0,
        )

    def test_equatorial_color_match_reduces_offset_and_preserves_outside(self):
        height, width = 180, 360
        north = torch.full((1, 3, height, width), 0.8)
        south = torch.full_like(north, 0.2)
        latitude = 90.0 - (torch.arange(height).float() + 0.5) * (180.0 / height)
        latitude = latitude[:, None].expand(height, width)
        hard_cut = torch.where(
            (latitude >= 0)[None, None], north, south
        )
        matched = match_equatorial_low_frequency(
            north, south, hard_cut, latitude, match_degrees=6.0
        )

        split = height // 2
        original_jump = (hard_cut[..., split - 1, :] - hard_cut[..., split, :]).abs().mean()
        matched_jump = (matched[..., split - 1, :] - matched[..., split, :]).abs().mean()
        self.assertLess(float(matched_jump), float(original_jump) * 0.1)
        outside = (latitude.abs() >= 6.0)[None, None].expand_as(hard_cut)
        self.assertTrue(torch.equal(matched[outside], hard_cut[outside]))

    def test_equatorial_color_match_preserves_owner_high_frequency(self):
        height, width = 64, 128
        texture = (torch.arange(width) % 2).float().view(1, 1, 1, width) * 0.1
        texture = texture.expand(1, 3, height, width)
        north = texture + 0.6
        south = texture + 0.2
        latitude = 90.0 - (torch.arange(height).float() + 0.5) * (180.0 / height)
        latitude = latitude[:, None].expand(height, width)
        hard_cut = torch.where((latitude >= 0)[None, None], north, south)
        matched = match_equatorial_low_frequency(
            north, south, hard_cut, latitude, match_degrees=6.0
        )

        row = height // 2 - 1
        torch.testing.assert_close(
            matched[..., row, 1:] - matched[..., row, :-1],
            hard_cut[..., row, 1:] - hard_cut[..., row, :-1],
            atol=1e-6, rtol=0,
        )

    def test_equatorial_color_match_disable_and_circular_shift(self):
        height, width = 32, 64
        generator = torch.Generator().manual_seed(7)
        north = torch.rand(1, 3, height, width, generator=generator)
        south = torch.rand(1, 3, height, width, generator=generator)
        latitude = 90.0 - (torch.arange(height).float() + 0.5) * (180.0 / height)
        latitude = latitude[:, None].expand(height, width)
        hard_cut = torch.where((latitude >= 0)[None, None], north, south)
        self.assertTrue(torch.equal(
            match_equatorial_low_frequency(north, south, hard_cut, latitude, 0.0),
            hard_cut,
        ))

        matched = match_equatorial_low_frequency(
            north, south, hard_cut, latitude, 6.0
        )
        shift = 11
        shifted = match_equatorial_low_frequency(
            torch.roll(north, shift, -1), torch.roll(south, shift, -1),
            torch.roll(hard_cut, shift, -1), latitude, 6.0,
        )
        torch.testing.assert_close(shifted, torch.roll(matched, shift, -1))

    def test_equatorial_color_match_rejects_negative_width(self):
        source = torch.zeros(1, 3, 8, 16)
        latitude = torch.zeros(8, 16)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            match_equatorial_low_frequency(
                source, source, source, latitude, -1.0
            )

    def test_equatorial_residual_correction_restores_natural_gradient(self):
        height, width = 180, 360
        latitude = 90.0 - (torch.arange(height).float() + 0.5) * (180.0 / height)
        latitude = latitude[:, None].expand(height, width)
        ramp = torch.arange(height).float().view(1, 1, height, 1) * 0.001
        source = ramp.expand(1, 3, height, width).clone()
        source[..., height // 2 :, :] += 0.2
        split = height // 2
        expected = 0.001
        before = source[..., split, :] - source[..., split - 1, :]

        corrected = correct_equatorial_seam_residual(
            source, latitude, residual_degrees=2.0, blur_degrees=0.5
        )
        after = corrected[..., split, :] - corrected[..., split - 1, :]
        self.assertLess(
            float((after - expected).abs().mean()),
            float((before - expected).abs().mean()) * 0.2,
        )
        outside = (latitude.abs() >= 2.0)[None, None].expand_as(source)
        self.assertTrue(torch.equal(corrected[outside], source[outside]))
        north_delta = corrected[..., split - 1, :] - source[..., split - 1, :]
        south_delta = corrected[..., split, :] - source[..., split, :]
        torch.testing.assert_close(north_delta, -south_delta, atol=1e-6, rtol=0)

    def test_equatorial_residual_disable_and_circular_shift(self):
        height, width = 32, 64
        generator = torch.Generator().manual_seed(11)
        source = torch.rand(1, 3, height, width, generator=generator)
        latitude = 90.0 - (torch.arange(height).float() + 0.5) * (180.0 / height)
        latitude = latitude[:, None].expand(height, width)
        self.assertTrue(torch.equal(
            correct_equatorial_seam_residual(source, latitude, 0.0), source
        ))

        corrected = correct_equatorial_seam_residual(
            source, latitude, residual_degrees=6.0, blur_degrees=0.5
        )
        shift = 9
        shifted = correct_equatorial_seam_residual(
            torch.roll(source, shift, -1), latitude,
            residual_degrees=6.0, blur_degrees=0.5,
        )
        torch.testing.assert_close(shifted, torch.roll(corrected, shift, -1))

    def test_equatorial_residual_rejects_invalid_blur(self):
        source = torch.zeros(1, 3, 8, 16)
        latitude = 90.0 - (torch.arange(8).float() + 0.5) * (180.0 / 8)
        latitude = latitude[:, None].expand(8, 16)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            correct_equatorial_seam_residual(
                source, latitude, residual_degrees=2.0, blur_degrees=0.0
            )


if __name__ == "__main__":
    unittest.main()
