import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from training.data import (
    ErpRecord,
    RejectedErp,
    discover_images,
    inspect_erp,
    split_for_scene,
    write_manifests,
)


class A1DataTests(unittest.TestCase):
    def test_scene_variants_stay_in_one_split(self):
        self.assertEqual(split_for_scene("scene-7"), split_for_scene("scene-7"))
        self.assertIn(split_for_scene("scene-7"), {"train", "validation", "test"})

    def test_scan_validates_erp_and_groups_subdirectory_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "living_room"
            scene.mkdir()
            valid = scene / "exposure_a.png"
            invalid = root / "flat.png"
            Image.new("RGB", (64, 32), (10, 20, 30)).save(valid)
            Image.new("RGB", (32, 32)).save(invalid)
            paths = discover_images(root, [".png"])
            self.assertEqual(paths, [invalid, valid])
            valid_result = inspect_erp(valid, root, 16)
            invalid_result = inspect_erp(invalid, root, 16)
            self.assertIsInstance(valid_result, ErpRecord)
            self.assertEqual(valid_result.scene_id, "living_room")
            self.assertIsInstance(invalid_result, RejectedErp)

    def test_manifests_are_jsonl_and_do_not_modify_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (64, 32)).save(source)
            before = source.read_bytes()
            record = inspect_erp(source, root, 16)
            output = root / "manifests"
            write_manifests([record], [], output)
            manifest = output / f"{record.split}.jsonl"
            parsed = json.loads(manifest.read_text().strip())
            self.assertEqual(parsed["scene_id"], "source")
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
