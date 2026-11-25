"""Geracao de vetores por imagem de origem e cenario."""
import hashlib
import numpy as np
from preparacao_dados.cifake import preprocess_cifake
from experimentos.transformacoes import CENARIOS, transformar
from extracao_caracteristicas import espaciais, espectrais

def gerar(linhas, raiz):
    resultado=[]
    for linha in linhas:
        caminho=raiz / linha['local_path']
        if hashlib.sha256(caminho.read_bytes()).hexdigest()!=linha['sha256']: raise ValueError('Hash da imagem divergente')
        original=preprocess_cifake(caminho)
        for cenario in CENARIOS:
            img=transformar(original,cenario)
            v=np.concatenate((espaciais.extrair(img),espectrais.extrair(img)))
            if v.shape!=(43,) or not np.isfinite(v).all(): raise ValueError('Vetor invalido')
            resultado.append({'image_id':linha['image_id'],'class':int(linha['class']),'experimental_split':linha['experimental_split'],'cenario':cenario,'vetor':v})
    return resultado
