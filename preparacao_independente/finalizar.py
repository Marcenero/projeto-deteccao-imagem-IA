"""Finalizacao deterministica da amostra revisada."""
import csv,hashlib,json,shutil
from collections import Counter
from pathlib import Path
from PIL import Image
from preparacao_independente.preparar import BASE,MANIFESTO,HIST,SEED,ler,sha,selecionar,dhash

DESTINO=Path('data/preparados/cifake_teste_independente')
def escolher(candidatos,seed): return selecionar([r for r in candidatos if r['automobile'].strip().lower()=='sim'],60,seed)
def finalizar(destino=DESTINO):
    destino=Path(destino); protocolo=json.loads((BASE/'protocolo_avaliacao.json').read_text(encoding='utf-8')); info=json.loads((BASE/'relatorio_preparacao.json').read_text(encoding='utf-8'))
    if info['seed']!=SEED or sha(MANIFESTO)!=protocolo['manifesto_anterior_sha256']: raise ValueError('Seed ou manifesto anterior divergente')
    for modelo in protocolo['modelos'].values():
        if sha(modelo['arquivo'])!=modelo['sha256']: raise ValueError('Hash de modelo divergente')
    if destino.exists(): raise FileExistsError('Destino existente: nao sera sobrescrito')
    candidatos=ler(BASE/'candidatos.csv'); mapa=ler(BASE/'mapa_revisao.csv'); lookup={r['image_id']:r for r in candidatos}
    if len(lookup)!=len(candidatos) or len({r['image_id'] for r in mapa})!=len(mapa) or any(lookup[r['image_id']]['automobile'].strip() not in ('sim','nao','ambiguo') for r in mapa): raise ValueError('Revisao incompleta ou mapa invalido')
    aprovados=[r for r in candidatos if r['automobile'].strip().lower()=='sim']
    if any(r['image_id'] not in {x['image_id'] for x in mapa} for r in aprovados): raise ValueError('Confirmacao sem folha de revisao')
    escolhidos=escolher(candidatos,info['seed']); anteriores=ler(MANIFESTO); historicos=ler(HIST); ids={r['image_id'] for r in anteriores+historicos}; hashes={r['sha256'] for r in anteriores+historicos}
    if Counter(r['class'] for r in escolhidos)!={'0':60,'1':60} or len({r['sha256'] for r in escolhidos})!=120 or len({r['image_id'] for r in escolhidos})!=120: raise ValueError('Selecao invalida')
    anteriores_dhash=[dhash(MANIFESTO.parent/r['local_path']) for r in anteriores]; dhashes=[]
    for r in escolhidos:
        p=Path(r['source_path']); classe='REAL' if r['class']=='0' else 'FAKE'; esperado=(Path('data/raw/cifake/test')/classe).resolve()
        if r['original_split']!='test' or p.resolve().parent!=esperado or r['image_id']!=f'test/{classe}/{p.name}' or r['image_id'] in ids or r['sha256'] in hashes or sha(p)!=r['sha256']: raise ValueError('Origem, ID ou hash invalido')
        with Image.open(p) as im:
            if im.format!='JPEG' or im.size!=(32,32): raise ValueError('Imagem invalida')
        d=dhash(p)
        if any((d^x).bit_count()<=3 for x in anteriores_dhash+dhashes): raise ValueError('Par perceptual suspeito: '+r['image_id'])
        dhashes.append(d)
    destino.mkdir(parents=True); (destino/'originais').mkdir()
    linhas=[]
    for r in escolhidos:
        local='originais/'+r['sha256']+'.jpg'; shutil.copyfile(r['source_path'],destino/local)
        if sha(destino/local)!=r['sha256']: raise ValueError('Copia divergente')
        linhas.append({'image_id':r['image_id'],'local_path':local,'class':r['class'],'original_split':'test','experimental_split':'teste_independente','semantic_category':'automobile','semantic_source':'human_visual','sha256':r['sha256']})
    manifesto=destino/'manifest.csv'
    with manifesto.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=linhas[0].keys());w.writeheader();w.writerows(linhas)
    rel={'seed':info['seed'],'metodo_sorteio':'ordenacao por image_id e random.Random(f"43:{classe}").shuffle; 60 por classe','contagens':dict(Counter(r['class'] for r in linhas)),'hash_manifesto':sha(manifesto),'hash_candidatos_anotados':sha(BASE/'candidatos.csv'),'hash_mapa_revisao':sha(BASE/'mapa_revisao.csv'),'hash_manifesto_anterior':sha(MANIFESTO),'criterio_perceptual':'dHash 64 bits, distancia <=3, contra os selecionados e 300 anteriores; nenhum par','confirmadas_por_classe':dict(Counter(r['class'] for r in aprovados))}
    (BASE/'relatorio_finalizacao.json').write_text(json.dumps(rel,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print(json.dumps(rel,indent=2)); return rel
if __name__=='__main__': finalizar()
