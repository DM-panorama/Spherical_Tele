"""CPU regressions for the single supported panorama inference path."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from telestylepanorama_inference import _style_paths, stylize_panorama


class _FakeEngine:
    def __init__(self):
        self.calls = []

    def inference_with_hemisphere_rgb_hard_cut(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        size = (args[7], args[6])
        return Image.new("RGB", size, "red"), Image.new("RGB", size, "blue")


class PanoramaInferenceTests(unittest.TestCase):
    def test_style_directory_is_filtered_and_sorted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8)).save(root / "B.PNG")
            Image.new("RGB", (8, 8)).save(root / "a.jpg")
            (root / "notes.txt").write_text("ignored", encoding="utf-8")
            (root / "nested").mkdir()

            self.assertEqual(
                [path.name for path in _style_paths(root)], ["a.jpg", "B.PNG"]
            )

    def test_style_directory_must_contain_an_image(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "no supported images"):
                _style_paths(Path(directory))

    def test_only_supported_generation_path_is_routed_and_restored(self):
        engine = _FakeEngine()
        result = stylize_panorama(
            engine=engine,
            content=Image.new("RGB", (64, 32), "white"),
            style=Image.new("RGB", (16, 16), "green"),
            prompt="prompt",
            seed=123,
            steps=4,
            hemisphere_size=32,
        )

        self.assertEqual(result.size, (64, 32))
        self.assertEqual(len(engine.calls), 1)
        args, kwargs = engine.calls[0]
        self.assertEqual(args[1].size, (32, 32))
        self.assertEqual(args[2].size, (32, 32))
        self.assertEqual(kwargs["color_match_degrees"], 6.0)
        pixels = np.asarray(result)
        self.assertTrue(np.all(pixels[0] == (0, 0, 255)))
        self.assertTrue(np.all(pixels[15] == (255, 0, 0)))

    def test_content_must_be_two_to_one(self):
        with self.assertRaisesRegex(ValueError, "2:1 ERP"):
            stylize_panorama(
                engine=_FakeEngine(),
                content=Image.new("RGB", (48, 32)),
                style=Image.new("RGB", (16, 16)),
                prompt="prompt",
                seed=123,
                steps=4,
            )


if __name__ == "__main__":
    unittest.main()
