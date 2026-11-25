import csv
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from genimage_prep.images import preprocess_image
from genimage_prep.prepare import prepare
from genimage_prep.sources import Candidate, LocalSource


def image_bytes(value: int, size=(13, 7), image_format="PNG") -> bytes:
    image = Image.new("L", size, value)
    with BytesIO() as stream:
        image.save(stream, format=image_format)
        return stream.getvalue()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "Stable Diffusion V1.4"
        self.quotas = {"train": 4, "validation": 2, "test": 3}
        self.cross_split_duplicates = set()
        for index, split in enumerate(("val", "train")):
            for label, folder in enumerate(("nature", "ai")):
                directory = self.root / split / folder
                directory.mkdir(parents=True)
                image_format = "JPEG" if label == 0 else "PNG"
                suffix = "JPEG" if label == 0 else "png"
                for serial in range(20 if split == "train" else 3):
                    stem = f"n01234567_{serial}" if label == 0 else f"042_sdv4_{serial}"
                    data = image_bytes(60 * index + 30 * label + serial,
                                       image_format=image_format)
                    (directory / f"{stem}.{suffix}").write_bytes(data)
                    if split == "val":
                        self.cross_split_duplicates.add(data)
        # Obriga a inspeção de todas as cópias e de um arquivo inválido em um teste.
        self.train_real = self.root / "train" / "nature"
        (self.train_real / "repeated_eval.png").write_bytes(image_bytes(0, image_format="JPEG"))
        (self.train_real / "broken.png").write_bytes(b"invalid")

    def run_pipeline(self, name, seed=42):
        output = self.base / name
        report_path = self.base / f"{name}.json"
        report = prepare(LocalSource(self.root, "fixture-v1"), profile="fixture",
                         quotas=self.quotas, seed=seed, size=(256, 256), output=output,
                         summary_path=report_path)
        if report["status"] == "failed":
            return [], report
        with (output / "manifest.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), report)
        return rows, report

    def test_balance_and_original_partitions(self):
        rows, report = self.run_pipeline("balanced")
        counts = Counter((r["experimental_split"], r["class"]) for r in rows)
        self.assertEqual(counts, {(split, label): n for split, n in self.quotas.items()
                                  for label in ("0", "1")})
        for row in rows:
            original = "val" if row["experimental_split"] == "test" else "train"
            self.assertEqual(row["image_id"].split("/")[0], original)
            self.assertEqual(row["original_split"], "test" if original == "val" else "train")
        self.assertEqual(report["selected"]["total"], 18)
        self.assertEqual(report["selected"]["possible_class_differences"]["test"]
                         ["formats_total_variation"], 1.0)

    def test_exact_duplicates_invalid_images_and_original_bytes(self):
        # Limita a fonte a seis originais válidos para obrigar a ler os dez itens.
        for path in sorted(self.train_real.glob("n*.JPEG"))[6:]:
            path.unlink()
        for serial in range(2):
            (self.train_real / f"dup{serial}.JPEG").write_bytes(
                image_bytes(0, image_format="JPEG")
            )
        # Exigir uma imagem adicional garante que toda a entrada seja auditada.
        self.quotas["validation"] = 3
        rows, report = self.run_pipeline("short")
        self.assertFalse(rows)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["invalid_count"], 1)
        self.assertEqual(report["duplicates_count"], 3)
        self.assertFalse((self.base / "short" / "manifest.csv").exists())

        self.quotas["validation"] = 2
        rows, _ = self.run_pipeline("unique")
        self.assertEqual(len(rows), 18)
        self.assertEqual(len({r["sha256"] for r in rows}), len(rows))
        for row in rows:
            original = (self.root / row["image_id"]).read_bytes()
            copy = (self.base / "unique" / row["local_path"]).read_bytes()
            self.assertEqual(copy, original)
            self.assertEqual(row["sha256"], sha256(copy).hexdigest())
            if row["experimental_split"] != "test":
                self.assertNotIn(copy, self.cross_split_duplicates)

    def test_sampling_is_reproducible(self):
        first, _ = self.run_pipeline("one")
        second, _ = self.run_pipeline("two")
        changed, _ = self.run_pipeline("other_seed", seed=43)
        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertNotEqual({r["image_id"] for r in first}, {r["image_id"] for r in changed})

    def test_categories_only_when_identifiable(self):
        self.assertEqual(Candidate("train/nature/n01234567_4.JPEG", 0, None).category,
                         "n01234567")
        self.assertEqual(Candidate("val/ai/042_sdv4_4.png", 1, None).category, "sd14_index:042")
        self.assertEqual(Candidate("val/nature/ILSVRC2012_val_00000037.JPEG", 0, None)
                         .category, "")

    def test_preprocessing(self):
        # As bordas laterais devem ficar fora do recorte central quadrado.
        source = Image.new("L", (13, 7), 15)
        source.paste(80, (3, 0, 10, 7))
        with BytesIO() as stream:
            source.save(stream, format="PNG")
            data = stream.getvalue()
        input_path = self.base / "source.png"
        input_path.write_bytes(data)
        from_path = preprocess_image(input_path)
        from_bytes = preprocess_image(data)
        self.assertEqual(from_path.size, (256, 256))
        self.assertEqual(from_path.mode, "RGB")
        self.assertEqual(from_path.getpixel((128, 128)), (80, 80, 80))
        self.assertGreater(from_path.getpixel((0, 128))[0], 40)
        self.assertGreater(from_path.getpixel((255, 128))[0], 40)
        self.assertEqual(from_path.tobytes(), from_bytes.tobytes())
        self.assertEqual(input_path.read_bytes(), data)
        with self.assertRaises(OSError):
            preprocess_image(b"invalid")


if __name__ == "__main__":
    unittest.main()
