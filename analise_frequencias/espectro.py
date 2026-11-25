"""Potência FFT e perfis radiais em ciclos por pixel."""
import numpy as np

EDGES = np.linspace(0, 0.5, 17)
BANDS = ((0.25, 0.3125), (0.3125, 0.375), (0.375, 0.5))


def spectrum(gray):
    gray = np.asarray(gray, dtype=np.float64)
    if gray.shape != (32, 32) or not np.isfinite(gray).all():
        raise ValueError("A entrada deve conter 32 × 32 valores finitos.")
    centered = gray - gray.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    freq = np.fft.fftshift(np.fft.fftfreq(32))
    radius = np.hypot(freq[:, None], freq[None, :])
    region = (radius > 0) & (radius <= 0.5)
    total = power[region].sum()
    counts = []
    radial_mean = []
    radial_energy = []
    if total <= 1e-24:
        return None
    normalized = np.where(region, power / total, 0)
    for index, (low, high) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        mask = region & (radius >= low) & ((radius <= high) if index == 15 else (radius < high))
        n = int(mask.sum())
        energy = float(normalized[mask].sum())
        counts.append(n)
        radial_energy.append(energy if n else None)
        radial_mean.append(energy / n if n else None)
    features = {}
    for index, (low, high) in enumerate(BANDS):
        mask = region & (radius >= low) & ((radius <= high) if index == 2 else (radius < high))
        features[f"E{index+1}"] = float(normalized[mask].sum())
    features["Etotal"] = float(normalized[region & (radius >= 0.25)].sum())
    features["Ebelow"] = float(normalized[region & (radius < 0.25)].sum())
    if not np.isclose(sum(features[f"E{i}"] for i in (1, 2, 3)), features["Etotal"], atol=1e-12):
        raise ValueError("As subfaixas não cobrem a faixa de interesse.")
    return {"power": power, "normalized": normalized, "radius": radius,
            "counts": counts, "radial_mean": radial_mean,
            "radial_energy": radial_energy, "features": features}
