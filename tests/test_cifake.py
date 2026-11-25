import csv
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
from genimage_prep.cifake import inventory, prepare, preprocess_cifake


class CifakeTests(unittest.TestCase):
    def test_inventory_sampling_and_semantic_selection(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            for split in ("train", "test"):
                for label, name in enumerate(("REAL", "FAKE")):
                    folder = root / split / name
                    folder.mkdir(parents=True)
                    for i in range(8):
                        color = (i * 20, label * 100, 50 if split == "train" else 150)
                        Image.new("RGB", (32, 32), color).save(folder / f"{i}.png")
            rows, invalid, summary = inventory(root)
            self.assertEqual(len(rows), 32)
            self.assertFalse(invalid)
            self.assertEqual({r["class"] for r in rows}, {0, 1})
            self.assertEqual({r["resolution"] for r in summary}, {"32x32"})
            selection = [{**r, "automobile": "sim" if not r["image_id"].endswith("7.png") else ""}
                         for r in rows]
            selection.append({**selection[0], "image_id": "duplicada"})
            quotas = {"train": 3, "validation": 1, "test": 2}
            a = prepare(selection, quotas, Path(tmp) / "a")
            b = prepare(selection, quotas, Path(tmp) / "b")
            self.assertEqual(a, b)
            self.assertEqual(len(a), 12)
            self.assertEqual(len({r["sha256"] for r in a}), 12)
            self.assertTrue(all(not r["image_id"].endswith("7.png") for r in a))
            for split, count in quotas.items():
                for label in (0, 1):
                    subset = [r for r in a if r["experimental_split"] == split and r["class"] == label]
                    self.assertEqual(len(subset), count)
                    self.assertTrue(all(r["original_split"] == ("test" if split == "test" else "train")
                                        for r in subset))
                    for row in subset:
                        image = Path(tmp) / "a" / row["local_path"]
                        self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(), row["sha256"])
                        self.assertEqual(preprocess_cifake(image).size, (32, 32))

    def test_rejects_insufficient_semantic_annotations(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                prepare([], {"train": 1, "validation": 1, "test": 1}, Path(tmp))


if __name__ == "__main__":
    unittest.main()
