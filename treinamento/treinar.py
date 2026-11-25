"""Treina e avalia tres florestas sem usar o teste."""
import csv, hashlib, json, platform, time
from pathlib import Path
import joblib, numpy as np, sklearn, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from sklearn.ensemble import RandomForestClassifier
from .modelos import carregar, NOMES, ARQUIVOS
from .avaliar import metricas

SAIDA=Path('resultados/treinamento')
def hash_arquivo(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def salvar_csv(p, rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    dados=carregar(); cfg=yaml.safe_load(Path('config.yaml').read_text(encoding='utf-8-sig'))['random_forest']
    (SAIDA/'modelos').mkdir(parents=True,exist_ok=True); (SAIDA/'figuras').mkdir(exist_ok=True)
    metricas_saida={'treino':[], 'validacao':[]}; predicoes=[]; importancias=[]; modelos={}; grupos={}
    for tipo,d in dados.items():
        tr=d['splits']=='train'; va=d['splits']=='validation'
        modelo=RandomForestClassifier(**cfg); inicio=time.perf_counter(); modelo.fit(d['X'][tr],d['y'][tr]); tempo=time.perf_counter()-inicio
        caminho=SAIDA/'modelos'/f'modelo_{tipo}.joblib'; joblib.dump(modelo,caminho); recarregado=joblib.load(caminho)
        if not np.array_equal(modelo.predict(d['X'][va]),recarregado.predict(d['X'][va])) or not np.allclose(modelo.predict_proba(d['X'][va]),recarregado.predict_proba(d['X'][va]),rtol=0,atol=1e-14): raise ValueError('Modelo recarregado divergente')
        for nome,mask in [('treino',tr),('validacao',va)]:
            yhat=modelo.predict(d['X'][mask]); m=metricas(d['y'][mask],yhat); metricas_saida[nome].append({'modelo':tipo,**m})
        yhat=modelo.predict(d['X'][va]); prob=modelo.predict_proba(d['X'][va])[:,list(modelo.classes_).index(1)]
        for i,real,prev,p in zip(np.array(d['ids'])[va],d['y'][va],yhat,prob): predicoes.append({'image_id':i,'modelo':tipo,'classe_verdadeira':int(real),'classe_prevista':int(prev),'probabilidade_fake':float(p),'acerto':int(real==prev)})
        for n,v in zip(NOMES[tipo],modelo.feature_importances_): importancias.append({'modelo':tipo,'caracteristica':n,'importancia':float(v)})
        if tipo=='hibrido': grupos={'espaciais':float(sum(modelo.feature_importances_[:40])),'espectrais':float(sum(modelo.feature_importances_[40:]))}
        modelos[tipo]={'tempo_segundos':tempo,'sha256':hash_arquivo(caminho),'arquivo':str(caminho),'caracteristicas':NOMES[tipo], 'recarregamento_identico':True}
        fig,ax=plt.subplots(figsize=(6,5)); mat=np.array(metricas_saida['validacao'][-1]['matriz_confusao']); im=ax.imshow(mat,vmin=0,vmax=15,cmap='Blues'); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['REAL','FAKE'],yticklabels=['REAL','FAKE'],xlabel='Prevista',ylabel='Verdadeira',title='Matriz de confusao - '+tipo)
        for (i,j),v in np.ndenumerate(mat): ax.text(j,i,str(v),ha='center',va='center')
        fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(SAIDA/'figuras'/f'matriz_{tipo}.png',dpi=150); plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,max(3,len(NOMES[tipo])*.21))); ordem=np.argsort(modelo.feature_importances_); ax.barh(np.array(NOMES[tipo])[ordem],modelo.feature_importances_[ordem]); ax.set(xlabel='Reducao media de impureza',title='Importancia - '+tipo,xlim=(0,1)); fig.tight_layout(); fig.savefig(SAIDA/'figuras'/f'importancia_{tipo}.png',dpi=150); plt.close(fig)
    for part,rows in metricas_saida.items():
        salvar_csv(SAIDA/f'metricas_{part}.csv',[{**r,'matriz_confusao':json.dumps(r['matriz_confusao'])} for r in rows])
    salvar_csv(SAIDA/'predicoes_validacao.csv',predicoes); salvar_csv(SAIDA/'importancia_caracteristicas.csv',importancias)
    nomes=['acuracia','precisao','recall','f1','especificidade']; fig,ax=plt.subplots(figsize=(9,4)); x=np.arange(len(nomes))
    for i,r in enumerate(metricas_saida['validacao']): ax.bar(x+(i-1)*.25,[r[n] for n in nomes],width=.25,label=r['modelo'])
    ax.set(xticks=x,xticklabels=nomes,ylim=(0,1),title='Metricas de validacao'); ax.legend(); fig.tight_layout(); fig.savefig(SAIDA/'figuras'/'comparacao_validacao.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for i,ax in enumerate(axes):
        for j,part in enumerate(('treino','validacao')): ax.bar(x+(j-.5)*.35,[metricas_saida[part][i][n] for n in nomes],width=.35,label=part)
        ax.set(xticks=x,xticklabels=nomes,ylim=(0,1),title=list(NOMES)[i]); ax.tick_params(axis='x',rotation=45)
    axes[0].legend(); fig.tight_layout(); fig.savefig(SAIDA/'figuras'/'treino_validacao.png',dpi=150); plt.close(fig)
    report={'parametros':cfg,'classes':{'0':'REAL','1':'FAKE'},'manifesto_sha256':hash_arquivo('data/preparados/cifake/manifest.csv'),'csv_sha256':{k:hash_arquivo(Path('resultados/caracteristicas')/v) for k,v in ARQUIVOS.items()},'versoes':{'python':platform.python_version(),'scikit_learn':sklearn.__version__,'joblib':joblib.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__},'modelos':modelos,'metricas':metricas_saida,'importancia_hibrida_por_grupo':grupos,'observacoes':'FAKE positiva; divisao por zero retorna 0; predict padrao desempata em favor da primeira classe (REAL); treino nao e avaliacao independente.'}
    (SAIDA/'relatorio.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print(json.dumps(metricas_saida,indent=2))
if __name__=='__main__': main()
