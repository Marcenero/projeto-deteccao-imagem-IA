"""Adaptador para a implementação espectral da Etapa 2."""
import numpy as np
from analise_frequencias.espectro import spectrum
from .espaciais import rgb_float, cinza

NOMES = ('E1', 'E2', 'E3')

def extrair(imagem):
    resultado = spectrum(cinza(rgb_float(imagem)))
    # Energia espectral nula não define uma distribuição normalizada.
    if resultado is None:
        return np.zeros(3, dtype=np.float64)
    return np.array([resultado['features'][n] for n in NOMES], dtype=np.float64)
