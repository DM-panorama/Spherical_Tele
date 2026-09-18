"""CPU regressions for the retained north/south chart geometry."""

import unittest

import numpy as np
import torch
from PIL import Image

from telestyle_spherical import (
    HemisphereLatentProjector,
    extract_stereographic_hemisphere,
)


class SphericalReprojectionTests(unittest.TestCase):
    def test_stereographic_hemisphere_has_valid_square_content(self):
        image = Image.new("RGB", (64, 32), (40, 80, 120))
        north = extract_stereographic_hemisphere(image, 32, 15.0, north=True)
        south = extract_stereographic_hemisphere(image, 32, 15.0, north=False)
        self.assertEqual(north.size, (32, 32))
        self.assertEqual(south.size, (32, 32))
        expected = torch.tensor([40, 80, 120], dtype=torch.uint8)
        self.assertTrue(torch.equal(
            torch.from_numpy(np.asarray(north).copy()[0, 0]), expected
        ))
        self.assertTrue(torch.equal(
            torch.from_numpy(np.asarray(south).copy()[-1, -1]), expected
        ))

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

    def test_hemisphere_sync_rejects_mismatched_charts(self):
        projector = HemisphereLatentProjector(16, 15.0, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "identical square shapes"):
            projector.synchronize(
                torch.zeros(1, 1, 16, 16), torch.zeros(1, 1, 8, 8)
            )


if __name__ == "__main__":
    unittest.main()
