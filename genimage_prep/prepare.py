"""Amostragem balanceada, verificação de integridade e relatório."""

from collections import Counter
from contextlib import closing
import csv
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from PIL import Image

from .images import inspect_image
from .sources import GENERATOR, HubSource, LocalSource


FIELDS = (
    "image_id", "local_path", "class", "origin", "original_split",
    "experimental_split", "semantic_category", "original_width",
    "original_height", "original_format", "sha256",
)
EXTENSIONS = {"JPEG": "jpg", "PNG": "png", "TIFF": "tif", "GIF": "gif",
              "BMP": "bmp", "WEBP": "webp"}
SPLITS = ("train", "validation", "test")
INVALID_IMAGE = (ValueError, OSError, SyntaxError, Image.DecompressionBombError)


def prepare(
    source: HubSource | LocalSource,
    *,
    profile: str,
    quotas: dict[str, int],
    seed: int,
    size: tuple[int, int],
    output: Path,
    summary_path: Path,
) -> dict:
    rows = []
    seen = {}
    invalid = []
    duplicates = []
    examined = Counter()
    sampled_groups = {"train": Counter(), "test": Counter()}
    streaming = source.provenance["mode"] == "streaming"
    shard_divisors = {"train": 4, "test": 2}
    summary = {
        "status": "failed",
        "dataset": {"name": "GenImage", "subset": GENERATOR, **source.provenance},
        "sampling": {
            "profile": profile,
            "seed": seed,
            "per_class": quotas,
            "method": "shuffled_shards_and_batches" if streaming else "shuffled_paths",
            "shard_quota_divisors": shard_divisors if streaming else None,
        },
        "preprocessing": {"size": list(size), "mode": "RGB", "orientation": "EXIF",
                          "crop": "center", "interpolation": "Lanczos"},
        "limitations": [
            "A integridade cobre candidatos verificados, com deduplicação por bytes.",
            "As categorias sd14_index e WNID não foram alinhadas; categorias ausentes ficam vazias.",
        ],
    }
    if streaming:
        summary["limitations"].append(
            "Streaming amostra lotes iniciais de shards embaralhados; as probabilidades "
            "de inclusão variam com a posição e a composição dos shards."
        )
    elif source.provenance["version"] is None:
        summary["limitations"].append("A versão do arquivo oficial extraído não foi fornecida.")

    try:
        output.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".prepare-", dir=output) as temporary:
            stage = Path(temporary)
            (stage / "originals").mkdir()
            for split in ("test", "train"):
                assignments = {
                    label: [
                        target for target in (("test",) if split == "test" else SPLITS[:2])
                        for _ in range(quotas[target])
                    ]
                    for label in (0, 1)
                }
                for label, targets in assignments.items():
                    Random(f"{seed}:{split}:{label}:assignment").shuffle(targets)
                per_group = {
                    label: ceil(len(targets) / shard_divisors[split]) if streaming else len(targets)
                    for label, targets in assignments.items()
                }

                for group, candidates in source.groups(split, seed):
                    selected = Counter()
                    with closing(candidates):
                        for candidate in candidates:
                            label = candidate.label
                            targets = assignments[label]
                            if not targets or selected[label] >= per_group[label]:
                                continue
                            examined[(split, label)] += 1
                            origin = {"image_id": candidate.image_id, "original_split": split,
                                      "class": label}
                            try:
                                contents = candidate.read()
                                digest = sha256(contents).hexdigest()
                                width, height, image_format = inspect_image(contents)
                            except INVALID_IMAGE as exc:
                                invalid.append({**origin, "reason": str(exc)})
                                continue
                            if digest in seen:
                                duplicates.append({**origin, "sha256": digest,
                                                   "first": seen[digest]})
                                continue

                            target = targets.pop()
                            seen[digest] = {**origin, "experimental_split": target}
                            filename = f"{digest}.{EXTENSIONS.get(image_format, 'img')}"
                            relative_path = Path("originals", filename)
                            (stage / relative_path).write_bytes(contents)
                            rows.append({
                                **origin,
                                "local_path": relative_path.as_posix(),
                                "origin": "ImageNet" if label == 0 else GENERATOR,
                                "experimental_split": target,
                                "semantic_category": candidate.category,
                                "original_width": width,
                                "original_height": height,
                                "original_format": image_format,
                                "sha256": digest,
                            })
                            selected[label] += 1
                            sampled_groups[split][group] += 1
                            if all(not pending or selected[key] >= per_group[key]
                                   for key, pending in assignments.items()):
                                break
                    if not any(assignments.values()):
                        break
                if any(assignments.values()):
                    missing = {label: len(targets) for label, targets in assignments.items()}
                    raise RuntimeError(f"Amostra insuficiente no split {split}: faltam {missing}.")

            rows.sort(key=lambda row: (SPLITS.index(row["experimental_split"]),
                                       row["class"], row["image_id"]))
            with (stage / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            originals = output / "originals"
            originals.mkdir(exist_ok=True)
            for image in (stage / "originals").iterdir():
                image.replace(originals / image.name)
            (stage / "manifest.csv").replace(output / "manifest.csv")
        summary.update(status="complete", manifest=str(output / "manifest.csv"))
    except (OSError, RuntimeError, ValueError) as exc:
        summary["error"] = str(exc)
    finally:
        summary.update({
            "stream_bytes_read": source.transfer_bytes,
            "candidates_examined": {
                split: {str(label): examined[(split, label)] for label in (0, 1)}
                for split in ("train", "test")
            },
            "groups_with_selected_rows": sampled_groups,
            "invalid_count": len(invalid),
            "duplicates_count": len(duplicates),
            "invalid_images": invalid,
            "duplicates": duplicates,
            "selected": summarize(rows),
        })
        write_summary(summary_path, summary)
    return summary


def summarize(rows: list[dict]) -> dict:
    distributions = {}
    for split in SPLITS:
        distributions[split] = {}
        for label in (0, 1):
            selected = [r for r in rows if r["experimental_split"] == split and r["class"] == label]
            distributions[split][str(label)] = {
                "count": len(selected),
                "resolutions": dict(Counter(
                    f"{r['original_width']}x{r['original_height']}" for r in selected
                )),
                "formats": dict(Counter(r["original_format"] for r in selected)),
                "categories": dict(Counter(r["semantic_category"] for r in selected
                                           if r["semantic_category"])),
                "unknown_category_count": sum(not r["semantic_category"] for r in selected),
            }

    comparisons = {}
    for split, classes in distributions.items():
        real, generated = classes["0"], classes["1"]
        if real["count"] and generated["count"]:
            comparisons[split] = {}
            for field in ("resolutions", "formats"):
                distance = sum(
                    abs(real[field].get(key, 0) / real["count"]
                        - generated[field].get(key, 0) / generated["count"])
                    for key in real[field].keys() | generated[field].keys()
                ) / 2
                comparisons[split][f"{field}_total_variation"] = round(distance, 6)
    return {"total": len(rows), "by_split_and_class": distributions,
            "possible_class_differences": comparisons}


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        staged.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
        staged.replace(path)
