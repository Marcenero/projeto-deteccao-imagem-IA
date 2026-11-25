import csv,json,time,platform
from collections import Counter
from pathlib import Path
import numpy as np, joblib, sklearn, scipy, PIL, yaml
from sklearn.ensemble import RandomForestClassifier
from preparacao_dados.cifake import preprocess_cifake
from experimentos.transformacoes import CENARIOS, transformar,PARAMETROS
from extracao_caracteristicas import NOMES_ESPACIAIS
from melhorias.espectro_ampliado import extrair,NOMES,FAIXAS,LIMITES,mascaras
from melhorias.treinamento_robusto import escrever,hash_arq,resumir
from treinamento.avaliar import metricas

RAIZ=Path('resultados/melhorias/espectro_ampliado'); ANTERIOR=Path('resultados/melhorias/aumento_dados'); MANIFESTO=Path('data/preparados/cifake/manifest.csv')
def carregar_csv(p):
    with p.open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))
def principal():
    inicio=time.perf_counter(); ant=json.loads((ANTERIOR/'relatorio.json').read_text(encoding='utf-8')); rel=json.loads(Path('resultados/treinamento/relatorio.json').read_text(encoding='utf-8')); mh=hash_arq(MANIFESTO)
    if mh!=ant['manifesto_sha256'] or mh!=rel['manifesto_sha256']: raise ValueError('Hash do manifesto divergente')
    if ant['parametros']!=rel['parametros'] or ant['parametros']!=yaml.safe_load(Path('config.yaml').read_text(encoding='utf-8-sig'))['random_forest'] or ant['transformacoes']!=PARAMETROS or ant['cenarios']!=list(CENARIOS): raise ValueError('Configuracao divergente')
    arquivos={}
    for k,info in rel['modelos'].items():
        arquivos[k+'_original']=Path(info['arquivo']); arquivos[k+'_aumentado']=ANTERIOR/'modelos'/f'modelo_{k}_aumentado.joblib'
        if hash_arq(arquivos[k+'_original'])!=info['sha256'] or hash_arq(arquivos[k+'_aumentado'])!=ant['modelos_aumentados'][k]: raise ValueError('Modelo anterior divergente')
    linhas=[r for r in carregar_csv(MANIFESTO) if r['experimental_split'] in ('train','validation')]
    if Counter((r['experimental_split'],r['class']) for r in linhas)!={('train','0'):105,('train','1'):105,('validation','0'):15,('validation','1'):15} or len({r['image_id'] for r in linhas})!=240: raise ValueError('Particoes divergentes')
    previos={}
    for part in ('train','validation'):
        nome='treino' if part=='train' else 'validacao'
        previos[part]={(r['image_id'],r['cenario']):r for r in carregar_csv(ANTERIOR/'caracteristicas'/f'{nome}.csv')}
    tempo=time.perf_counter(); dados=[]; zero=[]
    for linha in linhas:
        path=MANIFESTO.parent/linha['local_path']
        if hash_arq(path)!=linha['sha256']: raise ValueError('Imagem divergente')
        img=preprocess_cifake(path)
        for c in CENARIOS:
            anterior=previos[linha['experimental_split']][(linha['image_id'],c)]
            if anterior['class']!=linha['class'] or anterior['experimental_split']!=linha['experimental_split']: raise ValueError('Metadados divergentes')
            v,diag=extrair(transformar(img,c)); esp=np.array([float(anterior[n]) for n in NOMES_ESPACIAIS])
            if not np.allclose(v[:3],[float(anterior[n]) for n in NOMES[:3]],rtol=0,atol=1e-14): raise ValueError('Energias anteriores divergentes')
            if diag['energia_nula']: zero.append((linha['image_id'],c))
            dados.append({'image_id':linha['image_id'],'class':linha['class'],'experimental_split':linha['experimental_split'],'cenario':c,'B1':diag['B1'],**dict(zip(NOMES_ESPACIAIS,esp)),**dict(zip(NOMES,v))})
    tempo_extracao=time.perf_counter()-tempo
    for sub in ('modelos','caracteristicas','figuras'): (RAIZ/sub).mkdir(parents=True,exist_ok=True)
    tr=[r for r in dados if r['experimental_split']=='train']; va=[r for r in dados if r['experimental_split']=='validation']
    for nome,registros in [('treino',tr),('validacao',va)]: escrever(RAIZ/'caracteristicas'/f'{nome}.csv',registros)
    previstas=carregar_csv(ANTERIOR/'predicoes_validacao.csv'); anteriores=carregar_csv(ANTERIOR/'metricas_validacao.csv')
    esperado={(r['image_id'],r['cenario'],int(r['class'])) for r in va}
    if len(previstas)!=1260 or len(anteriores)!=42: raise ValueError('Saidas anteriores incompletas')
    for tipo in ('espacial','espectral','hibrido'):
        for regime in ('original','aumentado'):
            pred=[r for r in previstas if r['modelo']==tipo and r['regime']==regime]
            if len(pred)!=210 or {(r['image_id'],r['cenario'],int(r['classe_verdadeira'])) for r in pred}!=esperado: raise ValueError('Predicoes anteriores divergentes')
            for c in CENARIOS:
                ps=[r for r in pred if r['cenario']==c]; m=metricas([int(r['classe_verdadeira']) for r in ps],[int(r['classe_prevista']) for r in ps]); registro=next(r for r in anteriores if r['modelo']==tipo and r['regime']==regime and r['cenario']==c)
                if any(not np.isclose(m[k],float(registro[k]),atol=1e-14) for k in ('acuracia','precisao','recall','f1','especificidade')) or [int(registro[k]) for k in ('vn','fp','fn','vp')]!=np.array(m['matriz_confusao']).ravel().tolist(): raise ValueError('Metricas anteriores divergentes')
    met=[{**r,'representacao':'anterior'} for r in anteriores]; pred=[{**r,'representacao':'anterior'} for r in previstas]; novas=[]; importancias=[]; tempos={}; modelos={}
    for tipo in ('espectral','hibrido'):
        nomes=NOMES if tipo=='espectral' else NOMES_ESPACIAIS+NOMES
        for regime in ('original','aumentado'):
            subset=[r for r in tr if regime=='aumentado' or r['cenario']=='original']; X=np.array([[r[n] for n in nomes] for r in subset]); y=np.array([int(r['class']) for r in subset]); t=time.perf_counter()
            modelo=RandomForestClassifier(**ant['parametros']).fit(X,y); tempos[tipo+'_'+regime]=time.perf_counter()-t
            p=RAIZ/'modelos'/f'modelo_{tipo}_ampliado_{regime}.joblib'; joblib.dump(modelo,p); modelos[tipo+'_'+regime]={'sha256':hash_arq(p),'arquivo':str(p),'caracteristicas':nomes}
            importancias.extend({'modelo':tipo,'regime':regime,'caracteristica':n,'importancia':float(x)} for n,x in zip(nomes,modelo.feature_importances_))
            for c in CENARIOS:
                grupo=[r for r in va if r['cenario']==c]; Xv=np.array([[r[n] for n in nomes] for r in grupo]); yv=[int(r['class']) for r in grupo]; resultado=modelo.predict(Xv); prob=modelo.predict_proba(Xv)[:,list(modelo.classes_).index(1)]; m=metricas(yv,resultado); matriz=np.array(m.pop('matriz_confusao')).ravel()
                registro={'modelo':tipo,'regime':regime,'cenario':c,**m,**dict(zip(('vn','fp','fn','vp'),map(int,matriz))),'representacao':'ampliada'}; novas.append(registro); met.append(registro)
                pred.extend({'image_id':r['image_id'],'cenario':c,'modelo':tipo,'regime':regime,'classe_verdadeira':int(y0),'classe_prevista':int(yp),'probabilidade_fake':float(p0),'acerto':int(y0==yp),'representacao':'ampliada'} for r,y0,yp,p0 in zip(grupo,yv,resultado,prob))
    escrever(RAIZ/'metricas_validacao.csv',novas); escrever(RAIZ/'metricas_consolidadas.csv',met); escrever(RAIZ/'predicoes_validacao.csv',[r for r in pred if r['representacao']=='ampliada']); escrever(RAIZ/'predicoes_consolidadas.csv',pred); escrever(RAIZ/'importancia_caracteristicas.csv',importancias)
    resumo=[]
    for tipo,regime,rep in sorted({(r['modelo'],r['regime'],r['representacao']) for r in met}):
        conjunto=[{**r,'f1':float(r['f1'])} for r in met if (r['modelo'],r['regime'],r['representacao'])==(tipo,regime,rep)]; resumo.append({'modelo':tipo,'regime':regime,'representacao':rep,**resumir(conjunto)})
    escrever(RAIZ/'comparacao_modelos.csv',resumo)
    ab=ablar(resumo,pred); escrever(RAIZ/'ablacao.csv',ab); estatisticas(tr); figuras(tr,resumo,met,importancias,ab)
    if hash_arq(MANIFESTO)!=mh or any(hash_arq(p)!=(rel['modelos'][k.split('_')[0]]['sha256'] if k.endswith('_original') else ant['modelos_aumentados'][k.split('_')[0]]) for k,p in arquivos.items()): raise ValueError('Arquivos alterados')
    freq=np.fft.fftshift(np.fft.fftfreq(32)); r=np.hypot(freq[:,None],freq[None,:]); counts={k:int(x.sum()) for k,x in zip(FAIXAS,mascaras(r))}
    resultado={'manifesto_sha256':mh,'modelos_anteriores_sha256':{k:hash_arq(p) for k,p in arquivos.items()},'modelos_novos':modelos,'parametros':ant['parametros'],'transformacoes':PARAMETROS,'faixas':dict(zip(FAIXAS,zip(LIMITES[:-1],LIMITES[1:]))),'coeficientes':counts,'energia_nula':zero,'tempo_extracao_segundos':tempo_extracao,'tempo_treinamento_segundos':tempos,'tempo_total_segundos':time.perf_counter()-inicio,'contagens':{'treino':len(tr),'validacao':len(va),'novas_predicoes':len([x for x in pred if x['representacao']=='ampliada']),'combinacoes':len(met)},'versoes':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'pillow':PIL.__version__,'joblib':joblib.__version__}}
    (RAIZ/'relatorio.json').write_text(json.dumps(resultado,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print('Concluido',resultado['contagens'],resultado['tempo_total_segundos'])
def ablar(resumo,pred):
    pares=[]
    for tipo in ('espectral','hibrido'):
        for rep in ('anterior','ampliada'): pares.append((tipo,'original',rep,tipo,'aumentado',rep,'aumento_dados'))
        for regime in ('original','aumentado'): pares.append((tipo,regime,'anterior',tipo,regime,'ampliada','ampliacao_espectral'))
    for regime in ('original','aumentado'):
        for rep in ('anterior','ampliada'): pares.append(('espacial',regime,'anterior','hibrido',regime,rep,'comparacao_espacial'))
    out=[]
    for ta,ra,pa,tb,rb,pb,efeito in pares:
        a=next(x for x in resumo if (x['modelo'],x['regime'],x['representacao'])==(ta,ra,pa)); b=next(x for x in resumo if (x['modelo'],x['regime'],x['representacao'])==(tb,rb,pb))
        for c in CENARIOS:
            aa={r['image_id']:(int(r['classe_verdadeira']),int(r['classe_prevista'])) for r in pred if (r['modelo'],r['regime'],r['representacao'],r['cenario'])==(ta,ra,pa,c)}; bb={r['image_id']:(int(r['classe_verdadeira']),int(r['classe_prevista'])) for r in pred if (r['modelo'],r['regime'],r['representacao'],r['cenario'])==(tb,rb,pb,c)}
            if len(aa)!=30 or aa.keys()!=bb.keys() or any(aa[i][0]!=bb[i][0] for i in aa): raise ValueError('Pares invalidos')
            out.append({'efeito':efeito,'referencia':f'{ta}_{ra}_{pa}','comparado':f'{tb}_{rb}_{pb}','cenario':c,'delta_f1_original':b['f1_original']-a['f1_original'],'delta_f1_medio_transformado':b['f1_medio_transformados']-a['f1_medio_transformados'],'delta_pior_f1':b['pior_f1']-a['pior_f1'],'acerto_para_erro':sum(aa[i][0]==aa[i][1] and bb[i][0]!=bb[i][1] for i in aa),'erro_para_acerto':sum(aa[i][0]!=aa[i][1] and bb[i][0]==bb[i][1] for i in aa)})
    return out
def estatisticas(tr):
    nomes=('B1',)+NOMES; out=[]
    for classe in ('0','1'):
        for c in CENARIOS:
            grupo=[r for r in tr if r['class']==classe and r['cenario']==c]
            for n in nomes:
                x=np.array([r[n] for r in grupo]);out.append({'class':classe,'cenario':c,'caracteristica':n,'n':len(x),'media':x.mean(),'desvio':x.std(ddof=1),'mediana':np.median(x),'q1':np.quantile(x,.25),'q3':np.quantile(x,.75)})
    escrever(RAIZ/'estatisticas_treino.csv',out)
    grupo=[r for r in tr if r['cenario']=='original']; x=np.array([[r[n] for n in nomes] for r in grupo]); cor=np.corrcoef(x,rowvar=False)
    escrever(RAIZ/'correlacoes_treino.csv',[{'caracteristica':n,**{k:float(cor[i,j]) for j,k in enumerate(nomes)}} for i,n in enumerate(nomes)])

def figuras(tr,resumo,met,importancias,ab):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    def salvar(fig,n): fig.tight_layout(); fig.savefig(RAIZ/'figuras'/(n+'.png'),dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,2));
    for i,k in enumerate(FAIXAS): ax.barh(0,LIMITES[i+1]-LIMITES[i],left=LIMITES[i],label=k); ax.text((LIMITES[i+1]+LIMITES[i])/2,0,k,ha='center')
    ax.set(xlim=(0,.5),yticks=[],xlabel='Frequencia radial (ciclos/pixel)',title='Faixas espectrais'); salvar(fig,'faixas')
    fig,axes=plt.subplots(1,7,figsize=(15,4),sharey=True)
    for ax,n in zip(axes,FAIXAS):
        ax.boxplot([[r[n] for r in tr if r['cenario']=='original' and r['class']==c] for c in ('0','1')],tick_labels=['REAL','FAKE']); ax.set(title=n,ylim=(0,1))
    salvar(fig,'energias_classes')
    fig,axes=plt.subplots(1,2,figsize=(7,4))
    for ax,n in zip(axes,('centroide','entropia')): ax.boxplot([[r[n] for r in tr if r['cenario']=='original' and r['class']==c] for c in ('0','1')],tick_labels=['REAL','FAKE']); ax.set(title=n,ylim=(0,1))
    salvar(fig,'centroide_entropia')
    fig,axes=plt.subplots(2,4,figsize=(13,6))
    for ax,n in zip(axes.flat,NOMES):
        for cl in ('0','1'): ax.plot(CENARIOS,[np.mean([r[n] for r in tr if r['cenario']==c and r['class']==cl]) for c in CENARIOS],label='REAL' if cl=='0' else 'FAKE')
        ax.set(title=n,ylim=(0,1)); ax.tick_params(axis='x',rotation=70)
    axes[0,0].legend(); salvar(fig,'caracteristicas_transformacoes')
    fig,ax=plt.subplots(figsize=(11,5))
    for s in resumo:
        ax.plot(CENARIOS,[float(next(r['f1'] for r in met if (r['modelo'],r['regime'],r['representacao'],r['cenario'])==(s['modelo'],s['regime'],s['representacao'],c))) for c in CENARIOS],marker='.',label=s['modelo']+' '+s['regime']+' '+s['representacao'])
    ax.set(ylim=(0,1),title='F1 na validacao'); ax.tick_params(axis='x',rotation=30); ax.legend(fontsize=7); salvar(fig,'f1_cenarios')
    for n in ('f1_medio_transformados','pior_f1'):
        fig,ax=plt.subplots(figsize=(11,4)); ax.bar(range(len(resumo)),[r[n] for r in resumo]); ax.set(xticks=range(len(resumo)),xticklabels=[r['modelo']+'\n'+r['regime']+'\n'+r['representacao'] for r in resumo],ylim=(0,1),title=n); ax.tick_params(axis='x',labelsize=7); salvar(fig,n)
    for tipo in ('espectral','hibrido'):
        for regime in ('original','aumentado'):
            grupo=[r for r in importancias if r['modelo']==tipo and r['regime']==regime]; grupo.sort(key=lambda r:r['importancia']); fig,ax=plt.subplots(figsize=(8,max(3,len(grupo)*.18))); ax.barh([r['caracteristica'] for r in grupo],[r['importancia'] for r in grupo]); ax.set(xlim=(0,1),title='Importancia '+tipo+' '+regime); salvar(fig,'importancia_'+tipo+'_'+regime)
    grupo=[r for r in ab if r['cenario']=='original']; fig,ax=plt.subplots(figsize=(11,5)); ax.barh([r['comparado']+' vs '+r['referencia'] for r in grupo],[r['delta_f1_medio_transformado'] for r in grupo]); ax.axvline(0,color='black'); ax.set(xlim=(-1,1),title='Ablacao: variacao do F1 medio transformado'); salvar(fig,'ablacao')
if __name__=='__main__': principal()
