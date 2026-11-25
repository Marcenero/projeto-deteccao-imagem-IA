"""Avaliacao pareada dos modelos congelados no teste CIFAKE."""
import csv, json, hashlib, time
from collections import Counter
from pathlib import Path
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from preparacao_dados.cifake import preprocess_cifake
from extracao_caracteristicas import espaciais, espectrais, NOMES_HIBRIDOS
from treinamento.modelos import NOMES
from treinamento.avaliar import metricas
from .transformacoes import CENARIOS, PARAMETROS, transformar

MANIFESTO=Path('data/preparados/cifake/manifest.csv'); SAIDA=Path('resultados/robustez')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv_out(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def variacao(base, atual):
    return {k:atual[k]-base[k] for k in ('acuracia','precisao','recall','f1','especificidade')}
def parear(base,atual):
    if set(base)!=set(atual): raise ValueError('IDs divergentes')
    n=len(base)
    return {'acerto_para_erro':sum(base[i][0]==base[i][1] and atual[i][0]!=atual[i][1] for i in base), 'erro_para_acerto':sum(base[i][0]!=base[i][1] and atual[i][0]==atual[i][1] for i in base), 'taxa_mudanca':sum(base[i][1]!=atual[i][1] for i in base)/n}
def main():
    inicio=time.perf_counter(); SAIDA.mkdir(exist_ok=True,parents=True); (SAIDA/'figuras').mkdir(exist_ok=True)
    treinamento=json.loads(Path('resultados/treinamento/relatorio.json').read_text(encoding='utf-8')); mh=sha(MANIFESTO)
    if mh!=treinamento['manifesto_sha256']: raise ValueError('Manifesto divergente')
    modelos={}
    for k,info in treinamento['modelos'].items():
        if sha(info['arquivo'])!=info['sha256'] or list(NOMES[k])!=info['caracteristicas']: raise ValueError('Modelo ou ordem divergente')
        modelos[k]=joblib.load(info['arquivo'])
        if modelos[k].n_features_in_!=len(NOMES[k]): raise ValueError('Dimensao divergente')
    with MANIFESTO.open(encoding='utf-8',newline='') as f: rows=[r for r in csv.DictReader(f) if r['experimental_split']=='test']
    if len(rows)!=60 or len({r['image_id'] for r in rows})!=60 or Counter(r['class'] for r in rows)!={'0':30,'1':30}: raise ValueError('Teste inesperado')
    (SAIDA/'parametros.json').write_text(json.dumps({'cenarios':CENARIOS,'parametros':PARAMETROS},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    caracteristicas=[]; exemplos={}; arrays={c:[] for c in CENARIOS}
    for r in rows:
        p=MANIFESTO.parent/r['local_path']
        if sha(p)!=r['sha256']: raise ValueError('Imagem divergente')
        original=preprocess_cifake(p)
        for c in CENARIOS:
            img=transformar(original,c); v=np.concatenate((espaciais.extrair(img), espectrais.extrair(img)))
            if not np.isfinite(v).all() or len(v)!=43: raise ValueError('Caracteristicas invalidas')
            arrays[c].append(v); caracteristicas.append({'image_id':r['image_id'],'class':r['class'],'cenario':c,**dict(zip(NOMES_HIBRIDOS,v))})
            if r['class'] not in exemplos: exemplos[r['class']]={}
            if c not in exemplos[r['class']]: exemplos[r['class']][c]=np.array(img)
    csv_out(SAIDA/'caracteristicas_teste.csv',caracteristicas)
    y=np.array([int(r['class']) for r in rows]); predicoes=[]; metricas_linhas=[]; pareados={}
    for modelo,m in modelos.items():
        for c in CENARIOS:
            h=np.array(arrays[c]); x=h[:,:40] if modelo=='espacial' else h[:,40:] if modelo=='espectral' else h
            prev=m.predict(x); prob=m.predict_proba(x)[:,list(m.classes_).index(1)]; met=metricas(y,prev); mat=met.pop('matriz_confusao')
            metricas_linhas.append({'modelo':modelo,'cenario':c,**met,'vn':mat[0][0],'fp':mat[0][1],'fn':mat[1][0],'vp':mat[1][1]})
            pareados[(modelo,c)]={r['image_id']:(int(real),int(pred)) for r,real,pred in zip(rows,y,prev)}
            for r,real,pred,p in zip(rows,y,prev,prob): predicoes.append({'image_id':r['image_id'],'cenario':c,'modelo':modelo,'classe_verdadeira':int(real),'classe_prevista':int(pred),'probabilidade_fake':float(p),'acerto':int(real==pred)})
    csv_out(SAIDA/'metricas.csv',metricas_linhas); csv_out(SAIDA/'predicoes.csv',predicoes)
    diferencas=[]; resumos=[]
    for modelo in modelos:
        base=next(r for r in metricas_linhas if r['modelo']==modelo and r['cenario']=='original'); fs=[]
        for c in CENARIOS[1:]:
            atual=next(r for r in metricas_linhas if r['modelo']==modelo and r['cenario']==c); d=variacao(base,atual); fs.append(atual['f1'])
            diferencas.append({'modelo':modelo,'cenario':c,**{'variacao_'+k:v for k,v in d.items()},**parear(pareados[(modelo,'original')],pareados[(modelo,c)])})
        resumos.append({'modelo':modelo,'f1_original':base['f1'],'f1_medio_transformados':float(np.mean(fs)),'pior_f1':min(fs),'maior_queda_f1':max(0.0,base['f1']-min(fs))})
    csv_out(SAIDA/'variacoes_metricas.csv',diferencas); csv_out(SAIDA/'resumo_modelos.csv',resumos)
    gerar_figuras(metricas_linhas,diferencas,resumos,arrays,exemplos)
    report={'manifesto_sha256':mh,'modelos_sha256':{k:sha(v['arquivo']) for k,v in treinamento['modelos'].items()},'parametros':PARAMETROS,'cenarios':CENARIOS,'imagens':60,'predicoes':len(predicoes),'combinacoes':len(metricas_linhas),'tempo_segundos':time.perf_counter()-inicio,'classes':{'0':'REAL','1':'FAKE'},'comparacao':'pareada por image_id; referencia original do teste'}
    if sha(MANIFESTO)!=mh or any(sha(v['arquivo'])!=v['sha256'] for v in treinamento['modelos'].values()): raise ValueError('Arquivo alterado')
    (SAIDA/'relatorio.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print('Concluido:',report['tempo_segundos'],'segundos')
def gerar_figuras(ms,ds,resumos,arrays,exemplos):
    destino=SAIDA/'figuras'
    def salvar(fig,n): fig.tight_layout(); fig.savefig(destino/(n+'.png'),dpi=150); plt.close(fig)
    fig,ax=plt.subplots(2,7,figsize=(14,4))
    for i,cl in enumerate(('0','1')):
        for j,c in enumerate(CENARIOS): ax[i,j].imshow(exemplos[cl][c]); ax[i,j].axis('off'); ax[i,j].set_title(('REAL ' if i==0 else 'FAKE ')+c)
    salvar(fig,'exemplos')
    for met in ('acuracia','f1','recall'):
        fig,ax=plt.subplots(figsize=(9,4))
        for tipo in NOMES: ax.plot(CENARIOS,[next(r[met] for r in ms if r['modelo']==tipo and r['cenario']==c) for c in CENARIOS],marker='o',label=tipo)
        ax.set(ylim=(0,1),ylabel=met,title=met+' por cenario'); ax.legend(); ax.tick_params(axis='x',rotation=30); salvar(fig,met)
    fig,ax=plt.subplots(figsize=(12,5)); valores=np.array([[r[k] for k in ('acuracia','precisao','recall','f1','especificidade')] for r in ms]); im=ax.imshow(valores,vmin=0,vmax=1,aspect='auto',cmap='Blues'); ax.set(yticks=range(21),yticklabels=[r['modelo']+' / '+r['cenario'] for r in ms],xticks=range(5),xticklabels=['acuracia','precisao','recall','f1','especificidade'],title='Metricas por modelo e cenario'); fig.colorbar(im,ax=ax); salvar(fig,'mapa_metricas')
    fig,ax=plt.subplots(figsize=(9,4))
    for tipo in NOMES: ax.plot(CENARIOS[1:],[r['variacao_f1'] for r in ds if r['modelo']==tipo],marker='o',label=tipo)
    ax.axhline(0,color='black'); ax.set(ylim=(-1,1),title='Variacao de F1 em relacao ao original'); ax.legend(); ax.tick_params(axis='x',rotation=30); salvar(fig,'variacao_f1')
    for tipo in NOMES:
        fig,axes=plt.subplots(1,7,figsize=(14,3))
        for ax,c in zip(axes,CENARIOS):
            r=next(r for r in ms if r['modelo']==tipo and r['cenario']==c); mat=np.array([[r['vn'],r['fp']],[r['fn'],r['vp']]]); ax.imshow(mat,vmin=0,vmax=30,cmap='Blues'); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['R','F'],yticklabels=['R','F'],title=c)
            for (i,j),v in np.ndenumerate(mat): ax.text(j,i,str(v),ha='center',va='center')
        fig.suptitle('Matrizes de confusao: '+tipo+' (linhas verdadeiras)'); salvar(fig,'matrizes_'+tipo)
    fig,ax=plt.subplots(figsize=(9,4))
    for i,k in enumerate(('E1','E2','E3')): ax.plot(CENARIOS,[np.mean(np.array(arrays[c])[:,40+i]) for c in CENARIOS],marker='o',label=k)
    ax.set(ylim=(0,1),title='Energia espectral media no teste',ylabel='Energia normalizada'); ax.legend(); ax.tick_params(axis='x',rotation=30); salvar(fig,'energias_espectrais')
    fig,ax=plt.subplots(figsize=(6,4)); x=np.arange(3)
    ax.bar(x-.2,[r['f1_medio_transformados'] for r in resumos],.4,label='Media'); ax.bar(x+.2,[r['pior_f1'] for r in resumos],.4,label='Pior'); ax.set(xticks=x,xticklabels=[r['modelo'] for r in resumos],ylim=(0,1),title='F1 nos cenarios transformados'); ax.legend(); salvar(fig,'resumo_f1')

if __name__=='__main__': main()
