"""Análise exploratória pareada do treino CIFAKE."""
import csv
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from preparacao_dados.cifake import preprocess_cifake
from .espectro import EDGES, BANDS, spectrum

MANIFEST = Path("data/preparados/cifake/manifest.csv")
OUTPUT = Path("resultados/frequencias")
COLORS = {"0": "#176a8a", "1": "#c05238"}
NAMES = {"0": "REAL", "1": "FAKE"}
FEATURES = ("E1", "E2", "E3", "Etotal", "Ebelow")


def train_rows(rows):
    return [r for r in rows if r["experimental_split"] == "train"]


def gray(image):
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    return rgb @ np.array([0.299, 0.587, 0.114])


def jpeg90(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90, subsampling=0, optimize=False, progressive=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0, "mean": None, "std": None, "median": None,
                "q1": None, "q3": None, "min": None, "max": None}
    return {"n": int(a.size), "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if len(a) > 1 else None,
            "median": float(np.median(a)), "q1": float(np.quantile(a, .25)),
            "q3": float(np.quantile(a, .75)), "min": float(a.min()), "max": float(a.max())}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(data, features, examples):
    directory = OUTPUT / "figuras"
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 100})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / name, dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(7, 7))
    vmax = max(np.log1p(examples[k][1]["power"]).max() for k in ("0", "1"))
    for i, label in enumerate(("0", "1")):
        ax[i, 0].imshow(examples[label][0])
        ax[i, 0].set_title(NAMES[label] + " — imagem")
        ax[i, 1].imshow(np.log1p(examples[label][1]["power"]), cmap="magma", vmin=0, vmax=vmax,
                         extent=[-.5, .46875, -.5, .46875], origin="lower")
        ax[i, 1].set_title(NAMES[label] + " — log(1 + potência)")
        ax[i, 1].set_xlabel("fx (ciclos/pixel)")
        ax[i, 1].set_ylabel("fy (ciclos/pixel)")
        ax[i, 0].axis("off")
    save(fig, "01_exemplos_fft.png")

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    means = [np.mean([x["normalized"] for x in data[("baseline", label)]], axis=0) for label in ("0", "1")]
    vmax = max(m.max() for m in means)
    for axis, label, matrix in zip(ax, ("0", "1"), means):
        im = axis.imshow(matrix, cmap="magma", vmin=0, vmax=vmax,
                         extent=[-.5, .46875, -.5, .46875], origin="lower")
        axis.set_title(f"{NAMES[label]} (n={len(data[('baseline', label)])})")
        axis.set_xlabel("fx (ciclos/pixel)")
        axis.set_ylabel("fy (ciclos/pixel)")
    fig.colorbar(im, ax=ax.ravel().tolist(), label="Potência normalizada linear", shrink=.75)
    fig.savefig(directory / "02_espectros_medios.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    centers = (EDGES[:-1] + EDGES[1:]) / 2
    fig, ax = plt.subplots(figsize=(7, 4))
    for label in ("0", "1"):
        arr = np.array([x["radial_mean"] for x in data[("baseline", label)]], dtype=float)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)
        ax.plot(centers, mean, color=COLORS[label], label=f"{NAMES[label]} (n={len(arr)})")
        ax.fill_between(centers, np.maximum(0, mean-sd), mean+sd, color=COLORS[label], alpha=.18)
    ax.axvspan(.25, .5, color="grey", alpha=.1)
    ax.set(xlabel="Frequência radial (ciclos/pixel)", ylabel="Potência normalizada média por coeficiente",
           title="Perfil radial baseline, média ± desvio padrão")
    ax.legend()
    save(fig, "03_perfis_radiais.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(4)
    for offset, label in ((-.18, "0"), (.18, "1")):
        subset = [r for r in features if r["scenario"] == "baseline" and r["class"] == label and r["valid"]]
        y = [np.mean([r[key] for r in subset]) for key in FEATURES[:4]]
        ax.bar(x+offset, y, width=.36, color=COLORS[label], label=f"{NAMES[label]} (n={len(subset)})")
    ax.set(xticks=x, xticklabels=FEATURES[:4], ylabel="Energia relativa", title="Energia por faixa — baseline")
    ax.legend()
    save(fig, "04_energias.png")

    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
    for axis, key in zip(axes, FEATURES[:4]):
        groups = [[r[key] for r in features if r["scenario"] == "baseline" and r["class"] == label and r["valid"]]
                  for label in ("0", "1")]
        box = axis.boxplot(groups, tick_labels=["REAL", "FAKE"], patch_artist=True)
        for patch, label in zip(box["boxes"], ("0", "1")):
            patch.set_facecolor(COLORS[label])
        axis.set_title(key)
        axis.set_ylabel("Energia relativa")
    fig.suptitle("Distribuição das energias — baseline (n=105 por classe)")
    save(fig, "05_boxplots.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    for label in ("0", "1"):
        subset = [r for r in features if r["scenario"] == "baseline" and r["class"] == label and r["valid"]]
        ax.scatter([r["E1"] for r in subset], [r["E3"] for r in subset],
                   s=16, alpha=.6, color=COLORS[label], label=f"{NAMES[label]} (n={len(subset)})")
    ax.set(xlabel="E1 — energia relativa", ylabel="E3 — energia relativa", title="E1 versus E3 — baseline")
    ax.legend()
    save(fig, "06_dispersao.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, label in zip(axes, ("0", "1")):
        for scenario, marker in (("baseline", "o"), ("jpeg90", "s")):
            subset = [r for r in features if r["scenario"] == scenario and r["class"] == label and r["valid"]]
            axis.plot(FEATURES[:4], [np.mean([r[k] for r in subset]) for k in FEATURES[:4]],
                      marker=marker, color=COLORS[label], linestyle="-" if scenario == "baseline" else "--",
                      label=f"{scenario} (n={len(subset)})")
        axis.set(title=NAMES[label], ylabel="Energia relativa média")
        axis.legend()
    fig.suptitle("Energias antes e após JPEG qualidade 90")
    save(fig, "07_jpeg90.png")


def main():
    start = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))
    rows = train_rows(all_rows)
    if Counter(r["class"] for r in rows) != {"0": 105, "1": 105}:
        raise ValueError("O treino deve conter 105 imagens por classe.")
    data = {(scenario, label): [] for scenario in ("baseline", "jpeg90") for label in ("0", "1")}
    features, profiles, examples, invalid = [], [], {}, []
    for row in rows:
        path = MANIFEST.parent / row["local_path"]
        if sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Hash divergente: " + row["image_id"])
        image = preprocess_cifake(path)
        for scenario, transformed in (("baseline", image), ("jpeg90", jpeg90(image))):
            result = spectrum(gray(transformed))
            base = {"image_id": row["image_id"], "class": row["class"], "split": "train",
                    "scenario": scenario, "valid": result is not None}
            if result is None:
                invalid.append(base)
                features.append({**base, **{key: None for key in FEATURES}})
                continue
            features.append({**base, **result["features"]})
            data[(scenario, row["class"])].append(result)
            if scenario == "baseline" and row["class"] not in examples:
                examples[row["class"]] = (image, result)
            for i in range(16):
                profiles.append({**base, "bin": i, "lower": EDGES[i], "upper": EDGES[i+1],
                                 "center": (EDGES[i]+EDGES[i+1])/2, "coefficient_count": result["counts"][i],
                                 "mean_power": result["radial_mean"][i], "integrated_energy": result["radial_energy"][i]})
    write_csv(OUTPUT / "caracteristicas.csv", features)
    if not profiles or any(not group for group in data.values()):
        raise ValueError("Não há espectros de energia positiva em todas as classes e cenários.")
    write_csv(OUTPUT / "perfis_radiais.csv", profiles)
    summaries, radial_summary = [], []
    for scenario in ("baseline", "jpeg90"):
        for label in ("0", "1"):
            subset = [r for r in features if r["scenario"] == scenario and r["class"] == label and r["valid"]]
            for key in FEATURES:
                summaries.append({"scenario": scenario, "class": label, "feature": key,
                                  **stats([r[key] for r in subset])})
            for i in range(16):
                selected = [r for r in profiles if r["scenario"] == scenario and r["class"] == label and r["bin"] == i]
                for measure in ("mean_power", "integrated_energy"):
                    vals = [r[measure] for r in selected if r[measure] is not None]
                    radial_summary.append({"scenario": scenario, "class": label, "bin": i,
                                           "lower": EDGES[i], "upper": EDGES[i+1],
                                           "center": (EDGES[i]+EDGES[i+1])/2,
                                           "coefficient_count": selected[0]["coefficient_count"] if selected else 0,
                                           "measure": measure, **stats(vals)})
    write_csv(OUTPUT / "estatisticas_caracteristicas.csv", summaries)
    write_csv(OUTPUT / "estatisticas_radiais.csv", radial_summary)
    paired = {}
    for label in ("0", "1"):
        base = {r["image_id"]: r for r in features if r["class"] == label and r["scenario"] == "baseline" and r["valid"]}
        jpg = {r["image_id"]: r for r in features if r["class"] == label and r["scenario"] == "jpeg90" and r["valid"]}
        paired[label] = {key: stats([jpg[i][key] - base[i][key] for i in base.keys() & jpg.keys()]) for key in FEATURES}
    first = data[("baseline", "0")][0]
    radius = first["radius"]
    masks = [(radius >= low) & ((radius <= high) if i == 2 else (radius < high))
             for i, (low, high) in enumerate(BANDS)]
    report = {
        "dataset": "CIFAKE automobile", "manifest_sha256": sha256(MANIFEST.read_bytes()).hexdigest(),
        "analyzed_split": "train", "counts_per_class": dict(Counter(r["class"] for r in rows)),
        "scenarios": ["baseline", "jpeg90"], "zero_energy": invalid,
        "preprocessing": "RGB float64 / 255; grayscale 0.299 R + 0.587 G + 0.114 B; mean subtraction",
        "fft": "numpy fft2 and fftshift; abs(F)^2; no window; 0 < radius <= 0.5",
        "normalization": "sum of power in radial region", "radial_bins": 16,
        "band_edges": BANDS, "band_coefficient_counts": {f"E{i+1}": int(m.sum()) for i,m in enumerate(masks)},
        "radial_coefficient_counts": first["counts"],
        "jpeg": {"quality": 90, "subsampling": 0, "optimize": False, "progressive": False},
        "feature_statistics": summaries, "paired_jpeg_minus_baseline": paired,
        "class_mean_difference_fake_minus_real": {
            scenario: {
                key: next(r["mean"] for r in summaries if r["scenario"] == scenario and r["class"] == "1" and r["feature"] == key)
                     - next(r["mean"] for r in summaries if r["scenario"] == scenario and r["class"] == "0" and r["feature"] == key)
                for key in FEATURES
            }
            for scenario in ("baseline", "jpeg90")
        },
        "limitations": ["32x32 frequency resolution 1/32 cycles/pixel", "No window; edge leakage",
                        "Prior JPEG compression and CIFAKE construction may confound classes",
                        "JPEG recompression does not erase compression history or establish causality",
                        "Human selection assisted by a semantic classifier may introduce selection bias"],
        "seconds": time.perf_counter()-start,
    }
    plots(data, features, examples)
    report["seconds"] = time.perf_counter()-start
    (OUTPUT / "resumo.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Concluídas {len(rows)} imagens em {report['seconds']:.2f} s.")


if __name__ == "__main__":
    main()
