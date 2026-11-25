"""Priorização e revisão humana de automóveis no CIFAKE."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import yaml

from .cifake import RESULTS, preprocess_cifake, write_csv

REVISION = "786c16252c0fc58ee9adac063f8337cc4a7a497a"
SOURCE = Path(".venv") / f"pytorch-cifar-models-{REVISION}" / "pytorch_cifar_models/resnet.py"
WEIGHTS = Path(".venv/cifar10_resnet20-4118986f.pt")
SOURCE_HASH = "885b8fd2ccfb8ee27c02d3a382e2951622fdcbc2af2bbca375aaf71d40f5f246"
WEIGHTS_HASH = "4118986f0df73003d572b0e397f0ac7b3f60af1f31aff3d2da164536e36f6ec8"
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
AUTO = 1
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
PRED_FIELDS = ("image_id", "source_path", "original_split", "class", "top_category",
               "automobile_probability", "automobile_rank")
MAP_FIELDS = ("sheet", "position", "short_id", "image_id", "source_path", "group")
PRED_PATH = RESULTS / "predicoes_semanticas.csv"
MAP_PATH = RESULTS / "mapa_revisao_semantica.csv"
SELECTION = RESULTS / "selecao_categoria.csv"
SHEETS = RESULTS / "folhas_revisao_semantica"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def predictions_for_scores(rows, scores):
    if len(rows) != len(scores):
        raise ValueError("Quantidade de previsões diferente da quantidade de imagens.")
    output = []
    for row, score in zip(rows, scores):
        order = sorted(range(10), key=lambda i: (-float(score[i]), i))
        output.append({key: row[key] for key in ("image_id", "source_path", "original_split", "class")} | {
            "top_category": CLASSES[order[0]], "automobile_probability": float(score[AUTO]),
            "automobile_rank": order.index(AUTO) + 1})
    return output


def infer(rows, model_source=SOURCE, weights=WEIGHTS, batch_size=64):
    import torch
    if hashlib.sha256(model_source.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError("Código do modelo não corresponde à revisão fixada.")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != WEIGHTS_HASH:
        raise ValueError("Hash dos pesos diferente da versão fixada.")
    spec = importlib.util.spec_from_file_location("cifar_resnet_official", model_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.cifar10_resnet20(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    predictions = []
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            images = [np.asarray(preprocess_cifake(r["source_path"]), dtype=np.uint8).copy() for r in batch]
            tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div_(255)
            probabilities = model((tensor - mean) / std).softmax(dim=1).numpy()
            predictions.extend(predictions_for_scores(batch, probabilities))
            if (offset // batch_size) % 10 == 0:
                print(f"Inferência {min(offset + batch_size, len(rows))}/{len(rows)}", flush=True)
    return predictions, time.perf_counter() - start


def choose_review(predictions, seed=42, audit_per_group=25):
    ids = set()
    priority = []
    for row in predictions:
        if row["image_id"] in ids:
            raise ValueError("Identificador duplicado nas predições")
        ids.add(row["image_id"])
        if int(row["automobile_rank"]) <= 2:
            priority.append({**row, "group": "priority"})
    audit = []
    for split in ("train", "test"):
        for label in ("0", "1"):
            pool = sorted((r for r in predictions if int(r["automobile_rank"]) > 2
                           and r["original_split"] == split and str(r["class"]) == label),
                          key=lambda r: r["image_id"])
            random.Random(f"{seed}:audit:{split}:{label}").shuffle(pool)
            audit.extend({**r, "group": "audit"} for r in pool[:audit_per_group])
    random.Random(f"{seed}:priority").shuffle(priority)
    random.Random(f"{seed}:audit").shuffle(audit)
    return priority, audit


def make_sheets(predictions, seed=42, page_size=100):
    if MAP_PATH.exists():
        raise ValueError("Mapa de revisão já existe. Preserve a revisão antes de gerar novas folhas.")
    priority, audit = choose_review(predictions, seed)
    SHEETS.mkdir(parents=True, exist_ok=True)
    mapping = []
    number = 0
    for group in (priority, audit):
        for start in range(0, len(group), page_size):
            number += 1
            name = f"folha_{number:03}.png"
            canvas = Image.new("RGB", (1000, 1100), "white")
            draw = ImageDraw.Draw(canvas)
            for index, row in enumerate(group[start:start + page_size], 1):
                x, y = ((index - 1) % 10) * 100, ((index - 1) // 10) * 110
                thumb = preprocess_cifake(row["source_path"]).resize((80, 80), Image.Resampling.NEAREST)
                canvas.paste(thumb, (x + 10, y))
                draw.text((x + 10, y + 84), str(index), fill="black")
                mapping.append({"sheet": name, "position": index,
                                "short_id": f"{number:03}-{index:03}",
                                "image_id": row["image_id"], "source_path": row["source_path"],
                                "group": row["group"]})
            canvas.save(SHEETS / name)
    write_csv(MAP_PATH, mapping, MAP_FIELDS)
    return priority, audit, mapping


def parse_positions(value, max_position):
    if not value.strip():
        return set()
    values = {int(item) for item in value.replace(",", " ").split()}
    if any(n < 1 or n > max_position for n in values):
        raise ValueError("Número fora da folha.")
    return values


def apply_review(rows, sheet_rows, yes, ambiguous):
    if yes & ambiguous:
        raise ValueError("Uma imagem não pode ser aceita e ambígua simultaneamente.")
    lookup = {r["image_id"]: r for r in rows}
    for item in sheet_rows:
        row = lookup[item["image_id"]]
        if row["automobile"].strip():
            continue
        position = int(item["position"])
        row["automobile"] = "sim" if position in yes else "ambiguo" if position in ambiguous else "nao"
    return rows


def missing(rows, quotas):
    accepted = [r for r in rows if r["automobile"].strip().lower() in ("sim", "1")]
    counts = {}
    for split in ("train", "test"):
        quota = quotas["train"] + quotas["validation"] if split == "train" else quotas["test"]
        for label in ("0", "1"):
            hashes = {r["sha256"] for r in accepted if r["original_split"] == split and r["class"] == label}
            counts[f"{split}/{label}"] = max(0, quota - len(hashes))
    return counts


def review():
    selection = read_csv(SELECTION)
    mapping = read_csv(MAP_PATH)
    sheets = list(dict.fromkeys(row["sheet"] for row in mapping))
    quotas = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))["cifake"]["per_class"]
    lookup = {r["image_id"]: r for r in selection}
    for sheet in sheets:
        items = [r for r in mapping if r["sheet"] == sheet]
        if all(lookup[r["image_id"]]["automobile"].strip() for r in items):
            continue
        completed = sum(all(lookup[r["image_id"]]["automobile"].strip()
                            for r in mapping if r["sheet"] == name) for name in sheets)
        print(f"\nRevisadas {completed}/{len(sheets)}. Faltam por grupo {missing(selection, quotas)}")
        path = (SHEETS / sheet).resolve()
        print(f"Abra a imagem {path}")
        action = input("Enter para anotar, p para pular, q para sair: ").strip().lower()
        if action == "q":
            break
        if action == "p":
            continue
        try:
            yes = parse_positions(input("Automóveis de passeio (números separados por espaço): "), len(items))
            ambiguous = parse_positions(input("Ambíguos (números separados por espaço): "), len(items))
            apply_review(selection, items, yes, ambiguous)
        except ValueError as exc:
            print(f"Folha não salva: {exc}")
            continue
        temp = SELECTION.with_suffix(".tmp")
        write_csv(temp, selection, list(selection[0]))
        temp.replace(SELECTION)
        print("Folha salva. As demais imagens foram marcadas como não.")
    print("Faltam por grupo:", missing(selection, quotas))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("infer", "sheets", "review", "status"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    args = parser.parse_args()
    if args.action == "infer":
        if MAP_PATH.exists():
            raise SystemExit("Mapa de revisão existente. Preserve-o antes de alterar as predições.")
        rows = read_csv(SELECTION)
        predictions, seconds = infer(rows, args.source, args.weights)
        write_csv(PRED_PATH, predictions, PRED_FIELDS)
        weights_hash = hashlib.sha256(args.weights.read_bytes()).hexdigest()
        info = {"model": "cifar10_resnet20", "revision": REVISION,
                "source_sha256": SOURCE_HASH, "weights_sha256": weights_hash, "weights_file": args.weights.name,
                "seconds": seconds, "count": len(predictions), "batch_size": 64,
                "mean": MEAN, "std": STD, "device": "cpu",
                "classes": CLASSES}
        (RESULTS / "inferencia_semantica.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Inferência concluída em {seconds:.1f} s")
    elif args.action == "sheets":
        priority, audit, mapping = make_sheets(read_csv(PRED_PATH))
        print("Priorizados:", dict(Counter((r["original_split"], r["class"]) for r in priority)))
        print(f"Auditoria: {len(audit)}, folhas: {len({r['sheet'] for r in mapping})}")
    elif args.action == "review":
        review()
    else:
        cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))["cifake"]
        print("Faltam por grupo:", missing(read_csv(SELECTION), cfg["per_class"]))


if __name__ == "__main__":
    main()
