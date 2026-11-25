import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image
from preparacao_dados import selecao_semantica as semantic


class SemanticTests(unittest.TestCase):
    def sample(self, base):
        rows = []
        for i in range(12):
            image = base / f"{i}.png"
            Image.new("RGB", (32, 32), (i, i * 2, i * 3)).save(image)
            rows.append({"image_id": f"id{i}", "source_path": str(image),
                         "original_split": "train" if i < 8 else "test",
                         "class": str(i % 2), "sha256": str(i), "automobile": ""})
        return rows

    def test_automobile_index_prediction_and_order(self):
        self.assertEqual(semantic.CLASSES[semantic.AUTO], "automobile")
        rows = [{"image_id": "x", "source_path": "p", "original_split": "train", "class": "0"}]
        scores = [[0.2, 0.7] + [0.1 / 8] * 8]
        result = semantic.predictions_for_scores(rows, scores)
        self.assertEqual(result[0]["top_category"], "automobile")
        self.assertEqual(result[0]["automobile_rank"], 1)
        self.assertEqual(result[0]["automobile_probability"], 0.7)
        self.assertNotIn("automobile", result[0])
        with self.assertRaises(ValueError):
            semantic.predictions_for_scores(rows, [])

    def test_sheets_association_reproducibility_and_review(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.sample(root)
            predictions = [{**r, "automobile_rank": 2 if i < 8 else 5}
                           for i, r in enumerate(rows)]
            first, audit = semantic.choose_review(predictions, 42, 1)
            self.assertEqual((first, audit), semantic.choose_review(predictions, 42, 1))
            self.assertEqual(len(first), 8)
            self.assertEqual(len(audit), 2)
            with patch.object(semantic, "MAP_PATH", root / "map.csv"), \
                 patch.object(semantic, "SHEETS", root / "sheets"):
                _, _, mapping = semantic.make_sheets(predictions, 42, page_size=4)
                with (root / "map.csv").open(encoding="utf-8", newline="") as f:
                    self.assertEqual(list(csv.DictReader(f)), [
                        {k: str(v) for k, v in m.items()} for m in mapping])
                self.assertEqual(len({r["short_id"] for r in mapping}), len(mapping))
                lookup = {r["image_id"]: r["source_path"] for r in rows}
                self.assertTrue(all(lookup[r["image_id"]] == r["source_path"] for r in mapping))
                items = [r for r in mapping if r["sheet"] == mapping[0]["sheet"]]
                existing = next(r for r in rows if r["image_id"] == items[0]["image_id"])
                existing["automobile"] = "sim"
                semantic.apply_review(rows, items, {2}, {3})
                self.assertEqual(existing["automobile"], "sim")
                by_id = {r["image_id"]: r for r in rows}
                self.assertEqual(by_id[items[1]["image_id"]]["automobile"], "sim")
                self.assertEqual(by_id[items[2]["image_id"]]["automobile"], "ambiguo")
                self.assertEqual(by_id[items[3]["image_id"]]["automobile"], "nao")
                self.assertTrue(all(by_id[m["image_id"]]["automobile"] == ""
                                    for m in mapping if m["sheet"] != items[0]["sheet"]))
                with self.assertRaises(ValueError):
                    semantic.make_sheets(predictions)

    def test_counts_and_input_validation(self):
        self.assertEqual(semantic.parse_positions("1, 4 6", 6), {1, 4, 6})
        with self.assertRaises(ValueError):
            semantic.parse_positions("7", 6)
        quotas = {"train": 105, "validation": 15, "test": 30}
        self.assertEqual(semantic.missing([], quotas),
                         {"train/0": 120, "train/1": 120, "test/0": 30, "test/1": 30})


if __name__ == "__main__":
    unittest.main()
