"""Energias radiais ampliadas, centroide e entropia."""
import numpy as np
from analise_frequencias.espectro import spectrum
from extracao_caracteristicas.espaciais import rgb_float, cinza

NOMES=('E1','E2','E3','B2','B3','B4','centroide','entropia')
FAIXAS=('B1','B2','B3','B4','E1','E2','E3')
LIMITES=(0,.0625,.125,.1875,.25,.3125,.375,.5)
def mascaras(raio):
    regiao=(raio>0)&(raio<=.5)
    return tuple(regiao & (raio>=a) & ((raio<=b) if i==6 else (raio<b)) for i,(a,b) in enumerate(zip(LIMITES[:-1],LIMITES[1:])))
def entropia(p):
    p=np.asarray(p,dtype=float); positivos=p[p>0]
    return float(-np.sum(positivos*np.log(positivos))/np.log(p.size)) if positivos.size else 0.0

def extrair(imagem):
    resultado=spectrum(cinza(rgb_float(imagem)))
    if resultado is None: return np.zeros(8), {'B1':0.,'energia_nula':True}
    p=resultado['normalized']; r=resultado['radius']; masks=mascaras(r); region=(r>0)&(r<=.5)
    if not np.array_equal(np.sum(masks,axis=0),region.astype(int)): raise ValueError('Cobertura invalida')
    energias={k:float(p[m].sum()) for k,m in zip(FAIXAS,masks)}
    centroide=float((r[region]*p[region]).sum()); h=entropia(p[region])
    if not np.isclose(sum(energias.values()),1,atol=1e-12) or not -1e-12<=h<=1+1e-12: raise ValueError('Energia ou entropia invalida')
    for k in ('E1','E2','E3'):
        if not np.isclose(energias[k],resultado['features'][k],atol=1e-14,rtol=0): raise ValueError('Energia divergente')
    return np.array([*(energias[k] for k in NOMES[:6]),centroide,h]), {'B1':energias['B1'],'energia_nula':False}
