import unittest

import numpy as np
import torch
from PIL import Image

from telestyle_spherical import extract_stereographic_hemisphere

from training.geometry import (
    build_sphere_geometry,
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


if __name__ == "__main__":
    unittest.main()
