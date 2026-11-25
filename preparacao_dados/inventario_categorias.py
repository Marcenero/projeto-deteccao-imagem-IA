"""Inventário preliminar de nomes sem presumir correspondência semântica."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def inventory(class_map, train_index, test_index):
    real = Counter()
    synthetic = {"train": Counter(), "test": Counter()}
    unknown = Counter()
    for split, index in (("train", train_index), ("test", test_index)):
        for path, label in index["stable_diffusion_v_1_4"].items():
            filename = path.rsplit("/", 1)[-1]
            if label == 0:
                if split == "train":
                    real[filename.split("_")[0]] += 1
                else:
                    unknown[split] += 1
            elif label == 1:
                synthetic[split][filename.split("_")[0]] += 1
            else:
                raise ValueError(f"Classe inesperada: {label}")
    rows = []
    for number in range(1000):
        wnid, name = class_map[str(number)]
        for split in ("train", "test"):
            rows.append({
                "category": name, "wnid": wnid, "imagenet_index": number,
                "genimage_index_confirmed": "",
                "synthetic_prefix_hypothesis": f"{number:03d}",
                "real_count": real[wnid] if split == "train" else "",
                "synthetic_prefix_count": synthetic[split][f"{number:03d}"],
                "balanced_pairs_confirmed": "", "original_split": split,
                "selective_retrieval": "não validada",
            })
    return rows, dict(unknown)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--test-index", type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    rows, unknown = inventory(read(args.class_map), read(args.train_index), read(args.test_index))
    output = Path("resultados")
    output.mkdir(exist_ok=True)
    with (output / "inventario_categorias.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "incomplete",
        "revision": "3f4b9f921a673be09a93b335ed728cea0c6ecf33",
        "classes": 1000,
        "unidentified_real_evaluation_images": unknown,
        "limitations": [
            "Correspondência do prefixo sintético com o índice ImageNet não confirmada.",
            "A categoria das imagens reais de avaliação não está nos nomes do índice.",
            "Recuperação seletiva de registros Arrow não validada.",
        ],
        "sources": {
            "imagenet": "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
            "dataset": "https://huggingface.co/datasets/nebula/GenImage-arrow",
        },
    }
    (output / "resumo_inventario_categorias.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
