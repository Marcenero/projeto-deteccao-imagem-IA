"""Prepara candidatos CIFAKE para revisao humana, sem executar detectores."""
import argparse,csv,hashlib,json,random,time
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
from preparacao_dados import selecao_semantica as sem
from preparacao_dados.cifake import preprocess_cifake,write_csv

BASE=Path('resultados/melhorias/teste_independente'); RAIZ=Path('data/raw/cifake/test'); MANIFESTO=Path('data/preparados/cifake/manifest.csv'); HIST=Path('resultados/cifake/selecao_categoria.csv'); SEED=43

def ler(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def selecionar(elegiveis,n=1000,seed=SEED):
    saida=[]
    for cl in ('0','1'):
        grupo=sorted((r for r in elegiveis if r['class']==cl),key=lambda r:r['image_id']); random.Random(f'{seed}:{cl}').shuffle(grupo)
        if len(grupo)<n: raise ValueError('Candidatos insuficientes: '+cl)
        saida.extend(grupo[:n])
    return saida
def dhash(path):
    with Image.open(path) as im: a=np.asarray(im.convert('L').resize((9,8),Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits((a[:,1:]>a[:,:-1]).flatten()).tobytes(),'big')
def inventario():
    ant=ler(MANIFESTO); historico=ler(HIST); excluir_ids={r['image_id'] for r in historico}|{r['image_id'] for r in ant}; excluir_hash={r['sha256'] for r in historico}|{r['sha256'] for r in ant}; vistos=set(); rows=[]; cont=Counter()
    for nome,cl in (('REAL','0'),('FAKE','1')):
        for p in sorted((RAIZ/nome).glob('*.jpg')):
            image_id=f'test/{nome}/{p.name}'; cont['total_'+cl]+=1
            if image_id in excluir_ids: cont['id_excluido_'+cl]+=1; continue
            digest=sha(p)
            if digest in excluir_hash: cont['hash_historico_'+cl]+=1; continue
            if digest in vistos: cont['hash_duplicado_'+cl]+=1; continue
            vistos.add(digest); rows.append({'image_id':image_id,'source_path':p.as_posix(),'original_split':'test','class':cl,'sha256':digest,'automobile':''})
    return rows,dict(cont)
def preparar():
    if (BASE/'candidatos.csv').exists(): raise ValueError('Candidatos existentes; a revisao nao sera sobrescrita.')
    BASE.mkdir(parents=True,exist_ok=True); start=time.perf_counter(); elegiveis,cont=inventario(); candidatos=selecionar(elegiveis)
    write_csv(BASE/'candidatos.csv',candidatos,list(candidatos[0])); print('Elegiveis:',Counter(r['class'] for r in elegiveis),flush=True)
    pred,seconds=sem.infer(candidatos); write_csv(BASE/'predicoes_semanticas.csv',pred,sem.PRED_FIELDS)
    sem.MAP_PATH=BASE/'mapa_revisao.csv'; sem.SHEETS=BASE/'folhas_revisao'; prioridade,auditoria,mapa=sem.make_sheets(pred,seed=SEED)
    # Candidatos nao incluidos em folhas permanecem sem anotacao.
    anteriores=ler(MANIFESTO); vet_ant=[(r['image_id'],dhash(MANIFESTO.parent/r['local_path'])) for r in anteriores]
    vet=[(r['image_id'],dhash(r['source_path'])) for r in candidatos]; suspeitos=[]
    for i,(id_a,h) in enumerate(vet):
        for id_b,g in vet_ant:
            d=(h^g).bit_count()
            if d<=3:suspeitos.append({'id_a':id_a,'id_b':id_b,'origem_b':'amostra_anterior','distancia_dhash':d,'revisao':''})
        for id_b,g in vet[:i]:
            d=(h^g).bit_count()
            if d<=3:suspeitos.append({'id_a':id_a,'id_b':id_b,'origem_b':'candidatos','distancia_dhash':d,'revisao':''})
    fields=['id_a','id_b','origem_b','distancia_dhash','revisao'];
    with (BASE/'duplicatas_perceptuais_suspeitas.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(suspeitos)
    info={'seed':SEED,'metodo':'dHash 64 bits; distancia Hamming <= 3; suspeita, exige revisao humana','inventario':cont,'elegiveis':dict(Counter(r['class'] for r in elegiveis)),'candidatos':dict(Counter(r['class'] for r in candidatos)),'priorizados':dict(Counter(r['class'] for r in prioridade)),'auditoria':dict(Counter(r['class'] for r in auditoria)),'imagens_em_folhas':len(mapa),'folhas':len({r['sheet'] for r in mapa}),'pares_suspeitos':len(suspeitos),'tempo_inferencia_segundos':seconds,'tempo_total_segundos':time.perf_counter()-start,'manifesto_original_sha256':sha(MANIFESTO),'estado':'revisao_pendente','resnet_codigo_sha256':sem.SOURCE_HASH,'resnet_pesos_sha256':sem.WEIGHTS_HASH}
    (BASE/'relatorio_preparacao.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print(json.dumps(info,indent=2),flush=True)
def revisar():
    sem.SELECTION=BASE/'candidatos.csv';sem.MAP_PATH=BASE/'mapa_revisao.csv';sem.SHEETS=BASE/'folhas_revisao'
    rows=sem.read_csv(sem.SELECTION); mapa=sem.read_csv(sem.MAP_PATH); lookup={r['image_id']:r for r in rows}
    for folha in dict.fromkeys(r['sheet'] for r in mapa):
        itens=[r for r in mapa if r['sheet']==folha]
        if all(lookup[r['image_id']]['automobile'] for r in itens):continue
        print('\nAbra:',(sem.SHEETS/folha).resolve()); comando=input('Enter para anotar; p pular; q sair: ').strip().lower()
        if comando=='q':break
        if comando=='p':continue
        try:
            sim=sem.parse_positions(input('Automoveis de passeio: '),len(itens)); ambiguos=sem.parse_positions(input('Ambiguos: '),len(itens));sem.apply_review(rows,itens,sim,ambiguos)
        except ValueError as e:print('Folha nao salva:',e);continue
        tmp=sem.SELECTION.with_suffix('.tmp');write_csv(tmp,rows,list(rows[0]));tmp.replace(sem.SELECTION);print('Anotacoes salvas; demais marcadas nao.')
    status()
def status():
    rows=ler(BASE/'candidatos.csv'); print('Confirmadas por classe:',dict(Counter(r['class'] for r in rows if r['automobile']=='sim')),'de 60 por classe.'); pares=ler(BASE/'duplicatas_perceptuais_suspeitas.csv'); print('Pares perceptuais suspeitos:',len(pares),'- revise-os antes da selecao final.')
def main():
    p=argparse.ArgumentParser();p.add_argument('acao',choices=('preparar','revisar','status'));a=p.parse_args();{'preparar':preparar,'revisar':revisar,'status':status}[a.acao]()
if __name__=='__main__':main()
