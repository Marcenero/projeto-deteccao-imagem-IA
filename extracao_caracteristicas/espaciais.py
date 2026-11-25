"""Características espaciais calculadas sobre pixels RGB normalizados."""
import numpy as np
from PIL import Image

CANAIS = ('r', 'g', 'b')
NOMES = (tuple(f'media_{c}' for c in CANAIS) + tuple(f'desvio_{c}' for c in CANAIS)
         + tuple(f'hist_{c}_{i}' for c in CANAIS for i in range(8))
         + ('corr_rg', 'corr_rb', 'corr_gb')
         + tuple(f'dif_horizontal_{c}' for c in CANAIS)
         + tuple(f'dif_vertical_{c}' for c in CANAIS) + ('variancia_laplaciano',))

def rgb_float(imagem):
    if isinstance(imagem, Image.Image):
        a = np.asarray(imagem.convert('RGB'), dtype=np.float64) / 255.0
    else:
        original = np.asarray(imagem)
        a = original.astype(np.float64)
        if np.issubdtype(original.dtype, np.integer):
            a = a / 255.0
    if a.shape != (32, 32, 3) or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError('Esperado RGB 32 × 32 com valores finitos entre 0 e 1 (ou inteiros 0 a 255).')
    return a

def cinza(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114])

def extrair(imagem):
    a = rgb_float(imagem)
    valores = list(a.mean(axis=(0, 1))) + list(a.std(axis=(0, 1)))
    for i in range(3):
        valores.extend(np.histogram(a[:, :, i], bins=8, range=(0, 1))[0] / 1024)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        x, y = a[:, :, i].ravel(), a[:, :, j].ravel()
        x, y = x - x.mean(), y - y.mean()
        den = np.sqrt(np.dot(x, x) * np.dot(y, y))
        valores.append(float(np.dot(x, y) / den) if den > 1e-24 else 0.0)
    valores.extend(np.abs(np.diff(a, axis=1)).mean(axis=(0, 1)))
    valores.extend(np.abs(np.diff(a, axis=0)).mean(axis=(0, 1)))
    g = cinza(a)
    p = np.pad(g, 1, mode='edge')
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4*g
    valores.append(lap.var())
    return np.asarray(valores, dtype=np.float64)
