"""Inspeção, anotação visual e preparação do CIFAKE."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from random import Random
import shutil
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageOps
import yaml

from .imagens import inspect_image

ROOT = Path("data/raw/cifake")
RESULTS = Path("resultados/cifake")
LABELS = {"REAL": 0, "FAKE": 1}
FIELDS = ["image_id", "source_path", "original_split", "class", "sha256",
          "sheet", "position", "automobile"]


def preprocess_cifake(source):
    """Decodifica a imagem nativa e retorna RGB em 32 × 32."""
    with Image.open(source) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        if rgb.size != (32, 32):
            raise ValueError(f"Dimensões inesperadas: {rgb.size}")
        return rgb


def inventory(root=ROOT):
    rows, invalid = [], []
    details = Counter()
    for split in ("train", "test"):
        for name, label in LABELS.items():
            folder = root / split / name
            if not folder.is_dir():
                raise ValueError(f"Pasta ausente: {folder}")
            for path in sorted(p for p in folder.iterdir() if p.is_file()):
                try:
                    data = path.read_bytes()
                    width, height, fmt = inspect_image(data)
                except (OSError, ValueError, SyntaxError) as exc:
                    invalid.append({"path": str(path), "error": str(exc)})
                    continue
                digest = hashlib.sha256(data).hexdigest()
                row = {"image_id": f"{split}/{name}/{path.name}",
                       "source_path": path.as_posix(), "original_split": split,
                       "class": label, "sha256": digest}
                rows.append(row)
                details[(split, name, f"{width}x{height}", fmt)] += 1
    summary = [{"split": a, "class": b, "resolution": c, "format": d, "count": n}
               for (a, b, c, d), n in sorted(details.items())]
    return rows, invalid, summary


def inventory_archive(archive, root=ROOT):
    rows, invalid, details = [], [], Counter()
    with ZipFile(archive) as zipped:
        for info in sorted(zipped.infolist(), key=lambda entry: entry.filename):
            parts = Path(info.filename).parts
            if info.is_dir():
                continue
            if len(parts) != 3 or parts[0] not in ("train", "test") or parts[1] not in LABELS:
                continue
            split, name, filename = parts
            try:
                data = zipped.read(info)
                width, height, fmt = inspect_image(data)
            except (OSError, ValueError, SyntaxError) as exc:
                invalid.append({"path": info.filename, "error": str(exc)})
                continue
            rows.append({"image_id": info.filename, "source_path": root.joinpath(*parts).as_posix(),
                         "original_split": split, "class": LABELS[name],
                         "sha256": hashlib.sha256(data).hexdigest()})
            details[(split, name, f"{width}x{height}", fmt)] += 1
    summary = [{"split": a, "class": b, "resolution": c, "format": d, "count": n}
               for (a, b, c, d), n in sorted(details.items())]
    return rows, invalid, summary


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contacts(rows, seed=42, train_count=1500, test_count=500):
    chosen = []
    folder = RESULTS / "folhas_contato"
    folder.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for label in (0, 1):
            candidates = sorted((r for r in rows if r["original_split"] == split
                                 and r["class"] == label), key=lambda r: r["image_id"])
            Random(f"{seed}:{split}:{label}").shuffle(candidates)
            count = train_count if split == "train" else test_count
            for start in range(0, min(count, len(candidates)), 100):
                group = candidates[start:start + min(100, count - start)]
                sheet = f"{split}_{label}_{start // 100 + 1:03}.png"
                canvas = Image.new("RGB", (1000, 1100), "white")
                draw = ImageDraw.Draw(canvas)
                for i, row in enumerate(group):
                    x, y = (i % 10) * 100, (i // 10) * 110
                    with Image.open(row["source_path"]) as im:
                        thumb = im.convert("RGB").resize((80, 80), Image.Resampling.NEAREST)
                        canvas.paste(thumb, (x + 10, y))
                    draw.text((x + 10, y + 84), str(i + 1), fill="black")
                    chosen.append({**row, "sheet": sheet, "position": i + 1,
                                   "automobile": ""})
                canvas.save(folder / sheet)
    return chosen


def prepare(selection, quotas, output, seed=42):
    accepted = [row for row in selection if row["automobile"].strip().lower() in ("1", "sim")]
    seen = set()
    result = []
    for original_split, targets in (("test", ("test",)), ("train", ("train", "validation"))):
        for label in (0, 1):
            candidates = sorted((r for r in accepted if r["original_split"] == original_split
                                 and int(r["class"]) == label), key=lambda r: r["image_id"])
            Random(f"{seed}:{original_split}:{label}").shuffle(candidates)
            available = []
            candidate_hashes = set()
            for row in candidates:
                data = Path(row["source_path"]).read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if digest != row["sha256"]:
                    raise ValueError("Hash divergente: " + row["image_id"])
                inspect_image(data)
                preprocess_cifake(row["source_path"])
                if digest not in seen and digest not in candidate_hashes:
                    candidate_hashes.add(digest)
                    available.append(row)
            required = sum(quotas[target] for target in targets)
            if len(available) < required:
                raise ValueError(f"Anotações insuficientes para {original_split}/{label}: "
                                 f"{len(available)} disponíveis, {required} necessárias.")
            seen.update(row["sha256"] for row in available[:required])
            for target in targets:
                for row in available[:quotas[target]]:
                    result.append({**row, "experimental_split": target})
                available = available[quotas[target]:]
    output.mkdir(parents=True, exist_ok=True)
    originals = output / "originals"
    originals.mkdir(exist_ok=True)
    manifest = []
    for row in result:
        source = Path(row["source_path"])
        destination = originals / (row["sha256"] + source.suffix.lower())
        shutil.copyfile(source, destination)
        manifest.append({"image_id": row["image_id"], "local_path": destination.relative_to(output).as_posix(),
                         "class": row["class"], "original_split": row["original_split"],
                         "experimental_split": row["experimental_split"],
                         "semantic_category": "automobile", "semantic_source": "human_visual",
                         "sha256": row["sha256"]})
    write_csv(output / "manifest.csv", manifest, list(manifest[0]))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("inspect", "prepare"))
    parser.add_argument("--archive", type=Path, help="ZIP Kaggle já extraído em data/raw/cifake")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))["cifake"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.action == "inspect":
        selection_path = RESULTS / "selecao_categoria.csv"
        if selection_path.exists():
            raise SystemExit("A seleção já existe. Preserve as anotações antes de gerar novas folhas.")
        rows, invalid, summary = inventory_archive(args.archive) if args.archive else inventory()
        selected = contacts(rows, cfg["seed"], cfg["contact_candidates"]["train"],
                            cfg["contact_candidates"]["test"])
        write_csv(selection_path, selected, FIELDS)
        write_csv(RESULTS / "inventario_categorias.csv", summary,
                  ["split", "class", "resolution", "format", "count"])
        report = {
            "dataset": "CIFAKE", "kaggle_version": 3, "total_valid": len(rows),
            "invalid_count": len(invalid), "invalid": invalid, "distribution": summary,
            "selection_status": "awaiting_human_review", "selection_rows": len(selected),
            "seed": cfg["seed"], "semantic_labels": "not provided in directory structure",
            "duplicate_content_instances": len(rows) - len({r["sha256"] for r in rows}),
            "confirmed_semantic_count": 0,
        }
        if args.archive:
            with ZipFile(args.archive) as zipped:
                report["uncompressed_size_bytes"] = sum(entry.file_size for entry in zipped.infolist())
            report["archive_size_bytes"] = args.archive.stat().st_size
            with args.archive.open("rb") as archive:
                report["archive_sha256"] = hashlib.file_digest(archive, "sha256").hexdigest()
        (RESULTS / "resumo_dataset.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{len(rows)} válidas, {len(invalid)} inválidas, {len(selected)} para anotação.")
    else:
        with (RESULTS / "selecao_categoria.csv").open(encoding="utf-8", newline="") as f:
            selection = list(csv.DictReader(f))
        manifest = prepare(selection, cfg["per_class"], Path("data/preparados/cifake"), cfg["seed"])
        summary_path = RESULTS / "resumo_dataset.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["selection_status"] = "complete"
        summary["confirmed_selected"] = len(manifest)
        summary["experimental_counts"] = {
            split: {str(label): sum(r["experimental_split"] == split and
                                    int(r["class"]) == label for r in manifest)
                    for label in (0, 1)}
            for split in ("train", "validation", "test")
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
        print(f"{len(manifest)} imagens preparadas.")


if __name__ == "__main__":
    main()
