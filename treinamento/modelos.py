"""Carregamento estrito de caracteristicas e particoes."""
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from extracao_caracteristicas import NOMES_ESPACIAIS, NOMES_ESPECTRAIS, NOMES_HIBRIDOS

NOMES = {'espacial': NOMES_ESPACIAIS, 'espectral': NOMES_ESPECTRAIS, 'hibrido': NOMES_HIBRIDOS}
ARQUIVOS = {'espacial':'caracteristicas_espaciais.csv', 'espectral':'caracteristicas_espectrais.csv', 'hibrido':'caracteristicas_hibridas.csv'}
META = ('image_id', 'class', 'experimental_split')

def carregar(diretorio=Path('resultados/caracteristicas'), manifesto=Path('data/preparados/cifake/manifest.csv')):
    with Path(manifesto).open(encoding='utf-8',newline='') as f:
        manifest = [r for r in csv.DictReader(f) if r['experimental_split'] in ('train','validation')]
    referencia=[(r['image_id'],r['class'],r['experimental_split']) for r in manifest]
    if len(set(x[0] for x in referencia)) != 240 or Counter((x[2],x[1]) for x in referencia) != {('train','0'):105,('train','1'):105,('validation','0'):15,('validation','1'):15}: raise ValueError('Manifesto inesperado')
    saida={}
    for tipo, arquivo in ARQUIVOS.items():
        with (Path(diretorio)/arquivo).open(encoding='utf-8',newline='') as f:
            leitor=csv.DictReader(f)
            if leitor.fieldnames != list(META+NOMES[tipo]): raise ValueError('Colunas ou ordem inesperadas: '+tipo)
            rows=list(leitor)
        if [(r['image_id'],r['class'],r['experimental_split']) for r in rows] != referencia: raise ValueError('IDs, classes ou particoes divergentes')
        x=np.asarray([[float(r[n]) for n in NOMES[tipo]] for r in rows],dtype=np.float64)
        if not np.isfinite(x).all(): raise ValueError('Valor nao finito')
        saida[tipo]={'X':x,'y':np.array([int(r['class']) for r in rows]),'ids':[r['image_id'] for r in rows], 'splits':np.array([r['experimental_split'] for r in rows])}
    return saida
