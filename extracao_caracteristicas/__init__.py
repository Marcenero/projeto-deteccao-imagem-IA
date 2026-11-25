"""Extração reutilizável de características para imagens e arrays RGB."""
from . import espaciais, espectrais
import numpy as np

NOMES_ESPACIAIS = espaciais.NOMES
NOMES_ESPECTRAIS = espectrais.NOMES
NOMES_HIBRIDOS = NOMES_ESPACIAIS + NOMES_ESPECTRAIS

def extrair(imagem):
    espacial = espaciais.extrair(imagem)
    espectral = espectrais.extrair(imagem)
    return {'espaciais': espacial, 'espectrais': espectral,
            'hibridas': np.concatenate((espacial, espectral))}
