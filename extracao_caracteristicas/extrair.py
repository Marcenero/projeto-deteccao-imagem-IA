"""Extrai treino e validação, sem acessar imagens de teste."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import time
import numpy as np
from preparacao_dados.cifake import preprocess_cifake
from . import extrair, NOMES_ESPACIAIS, NOMES_ESPECTRAIS, NOMES_HIBRIDOS
from analise_frequencias.espectro import spectrum
from .espaciais import rgb_float, cinza

MANIFESTO = Path('data/preparados/cifake/manifest.csv')
SAIDA = Path('resultados/caracteristicas')
METADADOS = ('image_id', 'class', 'experimental_split')
NOMES = {'espaciais': NOMES_ESPACIAIS, 'espectrais': NOMES_ESPECTRAIS, 'hibridas': NOMES_HIBRIDOS}

def executar(manifesto=MANIFESTO, saida=SAIDA):
    inicio = time.perf_counter()
    manifesto, saida = Path(manifesto), Path(saida)
    hash_manifesto = sha256(manifesto.read_bytes()).hexdigest()
    with manifesto.open(encoding='utf-8', newline='') as f:
        linhas = [r for r in csv.DictReader(f) if r['experimental_split'] in ('train', 'validation')]
    contagens = Counter((r['experimental_split'], r['class']) for r in linhas)
    if contagens != {('train','0'):105, ('train','1'):105, ('validation','0'):15, ('validation','1'):15}:
        raise ValueError('Distribuição inesperada no manifesto.')
    ids = [r['image_id'] for r in linhas]
    if len(ids) != len(set(ids)):
        raise ValueError('Identificadores duplicados.')
    vetores = {k: [] for k in NOMES}
    sem_energia = []
    for linha in linhas:
        caminho = manifesto.parent / linha['local_path']
        if sha256(caminho.read_bytes()).hexdigest() != linha['sha256']:
            raise ValueError('Hash da imagem divergente: ' + linha['image_id'])
        imagem = preprocess_cifake(caminho)
        if spectrum(cinza(rgb_float(imagem))) is None:
            sem_energia.append(linha['image_id'])
        resultado = extrair(imagem)
        repetido = extrair(imagem)
        if any(not np.array_equal(resultado[tipo], repetido[tipo]) for tipo in NOMES):
            raise ValueError('Extra??o n?o reproduz?vel: ' + linha['image_id'])
        for tipo in NOMES:
            v = resultado[tipo]
            if len(v) != len(NOMES[tipo]) or not np.isfinite(v).all():
                raise ValueError('Vetor inválido: ' + linha['image_id'])
            vetores[tipo].append(v)
    saida.mkdir(parents=True, exist_ok=True)
    for tipo, nomes in NOMES.items():
        with (saida / f'caracteristicas_{tipo}.csv').open('w', encoding='utf-8', newline='') as f:
            w = csv.writer(f); w.writerow((*METADADOS, *nomes))
            for linha, vetor in zip(linhas, vetores[tipo]):
                w.writerow([*(linha[k] for k in METADADOS), *vetor])
    treino = np.array([v for r, v in zip(linhas, vetores['hibridas']) if r['experimental_split'] == 'train'])
    constantes = [NOMES_HIBRIDOS[i] for i in range(treino.shape[1]) if np.ptp(treino[:, i]) == 0]
    correlacoes = []
    for i in range(treino.shape[1]):
        for j in range(i + 1, treino.shape[1]):
            if NOMES_HIBRIDOS[i] in constantes or NOMES_HIBRIDOS[j] in constantes: continue
            c = float(np.corrcoef(treino[:, i], treino[:, j])[0, 1])
            if abs(c) >= .95: correlacoes.append({'caracteristicas': [NOMES_HIBRIDOS[i], NOMES_HIBRIDOS[j]], 'pearson': c})
    with (saida / 'estatisticas_treino.csv').open('w', encoding='utf-8', newline='') as f:
        w=csv.writer(f); w.writerow(['caracteristica','n','media','desvio_amostral','mediana','q1','q3','minimo','maximo'])
        for i,n in enumerate(NOMES_HIBRIDOS):
            a=treino[:,i]; w.writerow([n,len(a),a.mean(),a.std(ddof=1),np.median(a),np.quantile(a,.25),np.quantile(a,.75),a.min(),a.max()])
    historico = Path('resultados/frequencias/caracteristicas.csv')
    comparacao = {'disponivel': False}
    if historico.exists():
        with historico.open(encoding='utf-8', newline='') as f:
            referencia = {r['image_id']:r for r in csv.DictReader(f) if r['scenario']=='baseline' and r['valid']=='True'}
        erros = [abs(float(referencia[r['image_id']][nome])-float(v[i])) for r,v in zip(linhas,vetores['espectrais']) if r['experimental_split']=='train' and r['image_id'] in referencia for i,nome in enumerate(NOMES_ESPECTRAIS)]
        comparacao = {'disponivel': True, 'imagens_comparadas': len(erros)//3, 'erro_absoluto_maximo': max(erros) if erros else None}
        if len(erros) != 210*3 or max(erros) > 1e-12: raise ValueError('Divergência com a Etapa 2.')
    report = {'manifesto_sha256': hash_manifesto, 'imagens':len(linhas), 'contagens': {f'{p}_{c}':n for (p,c),n in sorted(contagens.items())}, 'dimensoes':{k:len(v) for k,v in NOMES.items()}, 'ordem':NOMES, 'metadados':METADADOS, 'parametros': {'rgb':'float64 / 255; 32x32', 'histograma':'8 intervalos fixos [0,1], normalizados por 1024', 'correlacao_canal_constante':0, 'laplaciano':'4 vizinhos; extensão de borda; cinza [0,1]; variância populacional', 'energia_nula':'vetor espectral zero; imagens registradas', 'correlacao_excessiva_abs_min':.95}, 'verificacoes':{'ids_unicos':True,'valores_finitos':True,'reproducibilidade_mesma_imagem':True,'particoes':['train','validation'],'correspondencia_arquivos':True,'energia_nula':sem_energia,'comparacao_etapa_2':comparacao}, 'constantes_treino':constantes, 'correlacoes_treino':correlacoes, 'tempo_segundos':time.perf_counter()-inicio}
    (saida/'relatorio.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(f"Extraídas {len(linhas)} imagens em {report['tempo_segundos']:.2f} s.")
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifesto',type=Path,default=MANIFESTO)
    parser.add_argument('--saida',type=Path,default=SAIDA)
    args=parser.parse_args(); executar(args.manifesto,args.saida)

if __name__ == '__main__': main()
