import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from telestyleimage_inference import BATCH_PANORAMA_CFG_SCALE, BATCH_PANORAMA_PROMPT, collect_content_images, stylize_batch_panorama


class ImageBatchTests(unittest.TestCase):
    def test_collect_content_images_filters_and_sorts_first_level_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            content_dir = Path(temp_dir)
            for name in ("z.jpg", "A.JPG", "middle.png", ".hidden.jpg", "notes.txt"):
                (content_dir / name).touch()
            (content_dir / "nested").mkdir()
            (content_dir / "nested" / "inside.jpg").touch()

            images = collect_content_images(content_dir)

            self.assertEqual([path.name for path in images], ["A.JPG", "middle.png", "z.jpg"])

    def test_collect_content_images_requires_existing_directory(self):
        with self.assertRaises(FileNotFoundError):
            collect_content_images("missing-content-directory")

    def test_stylize_batch_panorama_enables_seam_and_polar_fusion_defaults(self):
        engine = object()
        content = Image.new("RGB", (64, 32))
        style = Image.new("RGB", (64, 64))
        expected = Image.new("RGB", content.size)

        with patch("telestylepanorama_inference.stylize_panorama", return_value=(expected, 16, (96, 32))) as stylize:
            result = stylize_batch_panorama(engine, content, style)

        self.assertIs(result, expected)
        stylize.assert_called_once_with(
            engine=engine,
            content=content,
            style=style,
            prompt=BATCH_PANORAMA_PROMPT,
            seed=123,
            steps=4,
            margin_px=256,
            blend_px=96,
            enable_polar_fusion=True,
            polar_rotation_degrees=90.0,
            polar_blend_start_degrees=25.0,
            polar_blend_end_degrees=75.0,
            polar_fusion_steps=2,
            polar_fusion_strength=1.0,
            polar_lowpass_radius_latent=40,
            cfg_scale=BATCH_PANORAMA_CFG_SCALE,
        )


if __name__ == "__main__":
    unittest.main()
