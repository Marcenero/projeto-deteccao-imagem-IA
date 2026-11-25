"""Treinamento aumentado e comparacao exploratoria em validacao."""
import csv,hashlib,json,platform,time
from collections import Counter
from pathlib import Path
import joblib, numpy as np, sklearn, scipy, PIL, yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from treinamento.modelos import NOMES
from treinamento.avaliar import metricas
from extracao_caracteristicas import NOMES_HIBRIDOS
from experimentos.transformacoes import CENARIOS,PARAMETROS
from .aumento_dados import gerar

MANIFESTO=Path('data/preparados/cifake/manifest.csv'); SAIDA=Path('resultados/melhorias/aumento_dados')
def hash_arq(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def escrever(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def escolher(v,tipo): return v[:,:40] if tipo=='espacial' else v[:,40:] if tipo=='espectral' else v
def resumir(m):
    orig=next(r for r in m if r['cenario']=='original'); outros=[r['f1'] for r in m if r['cenario']!='original']
    return {'f1_original':orig['f1'],'f1_medio_transformados':float(np.mean(outros)),'pior_f1':min(outros),'maior_queda_f1':max(0.,orig['f1']-min(outros))}
def main():
    inicio=time.perf_counter(); rel=json.loads(Path('resultados/treinamento/relatorio.json').read_text(encoding='utf-8'))
    if hash_arq(MANIFESTO)!=rel['manifesto_sha256']: raise ValueError('Manifesto divergente')
    modelos_orig={}
    for tipo,info in rel['modelos'].items():
        if hash_arq(info['arquivo'])!=info['sha256'] or info['caracteristicas']!=list(NOMES[tipo]): raise ValueError('Modelo anterior divergente')
        modelos_orig[tipo]=joblib.load(info['arquivo'])
        if modelos_orig[tipo].n_features_in_!=len(NOMES[tipo]): raise ValueError('Dimensao divergente')
    with MANIFESTO.open(encoding='utf-8',newline='') as f: linhas=[r for r in csv.DictReader(f) if r['experimental_split'] in ('train','validation')]
    if Counter((r['experimental_split'],r['class']) for r in linhas)!={('train','0'):105,('train','1'):105,('validation','0'):15,('validation','1'):15} or len({r['image_id'] for r in linhas})!=240: raise ValueError('Particoes inesperadas')
    (SAIDA/'modelos').mkdir(parents=True,exist_ok=True); (SAIDA/'figuras').mkdir(exist_ok=True); (SAIDA/'caracteristicas').mkdir(exist_ok=True)
    t=time.perf_counter(); treino=gerar([r for r in linhas if r['experimental_split']=='train'],MANIFESTO.parent); validacao=gerar([r for r in linhas if r['experimental_split']=='validation'],MANIFESTO.parent); tempo_extracao=time.perf_counter()-t
    if len(treino)!=1470 or len(validacao)!=210 or Counter(r['class'] for r in treino)!={0:735,1:735}: raise ValueError('Instancias inesperadas')
    for nome,dados in [('treino',treino),('validacao',validacao)]:
        escrever(SAIDA/'caracteristicas'/f'{nome}.csv',[{k:r[k] for k in ('image_id','class','experimental_split','cenario')} | dict(zip(NOMES_HIBRIDOS,r['vetor'])) for r in dados])
    cfg=yaml.safe_load(Path('config.yaml').read_text(encoding='utf-8-sig'))['random_forest']
    if cfg!=rel['parametros']: raise ValueError('Hiperparametros diferem do treinamento original')
    X=np.array([r['vetor'] for r in treino]); y=np.array([r['class'] for r in treino]); modelos_novos={}; tempos={}
    for tipo in NOMES:
        t=time.perf_counter(); m=RandomForestClassifier(**cfg).fit(escolher(X,tipo),y); tempos[tipo]=time.perf_counter()-t
        p=SAIDA/'modelos'/f'modelo_{tipo}_aumentado.joblib'; joblib.dump(m,p); modelos_novos[tipo]=joblib.load(p)
    met=[]; pred=[]
    for tipo in NOMES:
        for regime,modelo in [('original',modelos_orig[tipo]),('aumentado',modelos_novos[tipo])]:
            for c in CENARIOS:
                registros=[r for r in validacao if r['cenario']==c]; x=escolher(np.array([r['vetor'] for r in registros]),tipo); yv=np.array([r['class'] for r in registros]); previsao=modelo.predict(x); prob=modelo.predict_proba(x)[:,list(modelo.classes_).index(1)]; m=metricas(yv,previsao); matriz=m.pop('matriz_confusao')
                met.append({'modelo':tipo,'regime':regime,'cenario':c,**m,'vn':int(matriz[0][0]),'fp':int(matriz[0][1]),'fn':int(matriz[1][0]),'vp':int(matriz[1][1])})
                pred.extend({'image_id':r['image_id'],'cenario':c,'modelo':tipo,'regime':regime,'classe_verdadeira':int(a),'classe_prevista':int(b),'probabilidade_fake':float(p),'acerto':int(a==b)} for r,a,b,p in zip(registros,yv,previsao,prob))
    escrever(SAIDA/'metricas_validacao.csv',met); escrever(SAIDA/'predicoes_validacao.csv',pred)
    comparacao=[]
    for tipo in NOMES:
        for regime in ('original','aumentado'):
            comparacao.append({'modelo':tipo,'regime':regime,**resumir([r for r in met if r['modelo']==tipo and r['regime']==regime])})
    escrever(SAIDA/'comparacao_modelos.csv',comparacao); figuras(met,comparacao,treino)
    for tipo,info in rel['modelos'].items():
        if hash_arq(info['arquivo'])!=info['sha256']: raise ValueError('Modelo anterior alterado')
    if hash_arq(MANIFESTO)!=rel['manifesto_sha256']: raise ValueError('Manifesto alterado')
    resultado={'manifesto_sha256':hash_arq(MANIFESTO),'modelos_originais':{k:v['sha256'] for k,v in rel['modelos'].items()},'modelos_aumentados':{k:hash_arq(SAIDA/'modelos'/f'modelo_{k}_aumentado.joblib') for k in NOMES},'parametros':cfg,'transformacoes':PARAMETROS,'cenarios':CENARIOS,'ordem_caracteristicas':NOMES,'contagens':{'treino_imagens':210,'treino_instancias':len(treino),'validacao_imagens':30,'validacao_instancias':len(validacao),'predicoes':len(pred),'combinacoes':len(met)},'tempo_extracao_segundos':tempo_extracao,'tempo_treinamento_segundos':tempos,'tempo_total_segundos':time.perf_counter()-inicio,'versoes':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'pillow':PIL.__version__,'joblib':joblib.__version__},'observacao':'Validacao exploratoria; instancias derivadas nao independentes; teste nao utilizado.'}
    (SAIDA/'relatorio.json').write_text(json.dumps(resultado,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print('Concluido:',resultado['contagens'],resultado['tempo_total_segundos'])
def figuras(met,comparacao,treino):
    def salvar(fig,n): fig.tight_layout(); fig.savefig(SAIDA/'figuras'/(n+'.png'),dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for tipo in NOMES:
        for regime in ('original','aumentado'):
            ax.plot(CENARIOS,[next(r['f1'] for r in met if r['modelo']==tipo and r['regime']==regime and r['cenario']==c) for c in CENARIOS],marker='o',linestyle='-' if regime=='aumentado' else '--',label=tipo+' '+regime)
    ax.set(ylim=(0,1),title='F1 na validacao por cenario',ylabel='F1'); ax.tick_params(axis='x',rotation=30); ax.legend(); salvar(fig,'f1_cenarios')
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,tipo in zip(axes,NOMES):
        for regime in ('original','aumentado'): ax.plot(CENARIOS,[next(r['f1'] for r in met if r['modelo']==tipo and r['regime']==regime and r['cenario']==c) for c in CENARIOS],marker='o',label=regime)
        ax.set(title=tipo,ylim=(0,1)); ax.tick_params(axis='x',rotation=45); ax.legend()
    salvar(fig,'comparacao_por_abordagem')
    for chave,nome in [('f1_medio_transformados','f1_medio'),('pior_f1','pior_f1')]:
        fig,ax=plt.subplots(figsize=(8,4)); ax.bar([r['modelo']+'\n'+r['regime'] for r in comparacao],[r[chave] for r in comparacao]); ax.set(ylim=(0,1),title=nome.replace('_',' '),ylabel='F1'); salvar(fig,nome)
    fig,ax=plt.subplots(figsize=(10,4))
    for tipo in NOMES:
        for regime in ('original','aumentado'):
            rows=[r for r in met if r['modelo']==tipo and r['regime']==regime]; base=next(r['f1'] for r in rows if r['cenario']=='original'); ax.plot(CENARIOS[1:],[next(r['f1'] for r in rows if r['cenario']==c)-base for c in CENARIOS[1:]],marker='o',label=tipo+' '+regime)
    ax.axhline(0,color='black'); ax.set(ylim=(-1,1),title='Variacao de F1 em relacao ao original'); ax.tick_params(axis='x',rotation=30); ax.legend(); salvar(fig,'variacoes_f1')
    for tipo in NOMES:
        fig,axes=plt.subplots(2,7,figsize=(14,5))
        for i,regime in enumerate(('original','aumentado')):
            for j,c in enumerate(CENARIOS):
                r=next(r for r in met if r['modelo']==tipo and r['regime']==regime and r['cenario']==c); a=np.array([[r['vn'],r['fp']],[r['fn'],r['vp']]]); axes[i,j].imshow(a,vmin=0,vmax=15,cmap='Blues'); axes[i,j].set_title(regime+'\n'+c); axes[i,j].set(xticks=[0,1],yticks=[0,1],xticklabels=['R','F'],yticklabels=['R','F'])
                for (u,v),x in np.ndenumerate(a): axes[i,j].text(v,u,str(x),ha='center',va='center')
        fig.suptitle('Matrizes de confusao - '+tipo+' (linhas verdadeiras)'); salvar(fig,'matrizes_'+tipo)
    fig,axes=plt.subplots(1,3,figsize=(10,4))
    for i,ax in enumerate(axes):
        a=[r['vetor'][40+i] for r in treino if r['cenario']=='original']; b=[r['vetor'][40+i] for r in treino]; ax.boxplot([a,b],tick_labels=['original','aumentado']); ax.set(title='E'+str(i+1),ylim=(0,1))
    fig.suptitle('Distribuicao das energias no treino'); salvar(fig,'energias_treino')
if __name__=='__main__': main()
