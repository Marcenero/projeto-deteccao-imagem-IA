"""Avaliacao final pareada dos dez modelos congelados."""
import csv,hashlib,json,time,platform
from pathlib import Path
from collections import Counter
import numpy as np,joblib,sklearn,PIL,scipy
from preparacao_independente.finalizar import DESTINO,escolher
from preparacao_independente.preparar import BASE,MANIFESTO,ler,sha
from preparacao_dados.cifake import preprocess_cifake
from experimentos.transformacoes import CENARIOS,transformar,PARAMETROS
from extracao_caracteristicas import espaciais,NOMES_ESPACIAIS,NOMES_ESPECTRAIS,NOMES_HIBRIDOS
from melhorias.espectro_ampliado import extrair as espectro_ampliado,NOMES as NOMES_AMPLIADOS
from treinamento.avaliar import metricas
from melhorias.treinamento_robusto import escrever,resumir
from experimentos.robustez import parear

SAIDA=BASE/'avaliacao'; CAMINHO=DESTINO/'manifest.csv'
def nomes_modelo(k):
    if k.startswith('espacial'): return NOMES_ESPACIAIS
    ampliado=k.endswith('_ampliado'); espectro=NOMES_AMPLIADOS if ampliado else NOMES_ESPECTRAIS
    return espectro if k.startswith('espectral') else NOMES_ESPACIAIS+espectro

def validar_entrada():
    protocolo=json.loads((BASE/'protocolo_avaliacao.json').read_text(encoding='utf-8')); final=json.loads((BASE/'relatorio_finalizacao.json').read_text(encoding='utf-8')); prep=json.loads((BASE/'relatorio_preparacao.json').read_text(encoding='utf-8'))
    if sha(CAMINHO)!=final['hash_manifesto'] or sha(MANIFESTO)!=protocolo['manifesto_anterior_sha256'] or sha(BASE/'candidatos.csv')!=final['hash_candidatos_anotados'] or sha(BASE/'mapa_revisao.csv')!=final['hash_mapa_revisao'] or tuple(protocolo['cenarios'])!=CENARIOS: raise ValueError('Integridade de manifesto, anotacoes ou cenarios divergente')
    rows=ler(CAMINHO); escolhidos=escolher(ler(BASE/'candidatos.csv'),prep['seed'])
    if [(r['image_id'],r['class'],r['sha256']) for r in rows]!=[(r['image_id'],r['class'],r['sha256']) for r in escolhidos] or Counter(r['class'] for r in rows)!={'0':60,'1':60}: raise ValueError('Selecao divergente')
    modelos={}
    if len(protocolo['modelos'])!=10: raise ValueError('Numero de modelos incorreto')
    for k,info in protocolo['modelos'].items():
        if sha(info['arquivo'])!=info['sha256'] or tuple(info['caracteristicas'])!=nomes_modelo(k): raise ValueError('Hash ou ordem divergente: '+k)
        m=joblib.load(info['arquivo'])
        if m.n_features_in_!=len(nomes_modelo(k)) or list(m.classes_)!=[0,1] or any(m.get_params()[n]!=v for n,v in protocolo['hiperparametros'].items()): raise ValueError('Modelo incompativel: '+k)
        modelos[k]=m
    for r in rows:
        if r['original_split']!='test' or sha(DESTINO/r['local_path'])!=r['sha256']: raise ValueError('Arquivo de amostra divergente')
    return protocolo,rows,modelos

def main():
    import argparse
    global SAIDA
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--saida', type=Path, default=SAIDA); SAIDA=parser.parse_args().saida
    inicio=time.perf_counter(); protocolo,rows,modelos=validar_entrada()
    if SAIDA.exists(): raise FileExistsError('Avaliacao ja existe; os resultados nao serao sobrescritos')
    SAIDA.mkdir(parents=True); (SAIDA/'figuras').mkdir()
    arrays={c:[] for c in CENARIOS}; caracteristicas=[]; t=time.perf_counter()
    for r in rows:
        img=preprocess_cifake(DESTINO/r['local_path'])
        for c in CENARIOS:
            imagem=transformar(img,c); espacial=espaciais.extrair(imagem); esp,diag=espectro_ampliado(imagem); vetor=np.concatenate((espacial,esp))
            if not np.isfinite(vetor).all() or len(vetor)!=48: raise ValueError('Vetor invalido')
            arrays[c].append(vetor); caracteristicas.append({'image_id':r['image_id'],'class':r['class'],'cenario':c,'B1':diag['B1'],**dict(zip(NOMES_ESPACIAIS+NOMES_AMPLIADOS,vetor))})
    tempo_extracao=time.perf_counter()-t; escrever(SAIDA/'caracteristicas.csv',caracteristicas)
    y=np.array([int(r['class']) for r in rows]); met=[]; pred=[]; mapas={}
    for nome,m in modelos.items():
        for c in CENARIOS:
            v=np.array(arrays[c]); x=v[:,:40] if nome.startswith('espacial') else v[:,40:] if nome.startswith('espectral') and nome.endswith('_ampliado') else v[:,40:43] if nome.startswith('espectral') else v if nome.endswith('_ampliado') else v[:,:43]
            yp=m.predict(x); probs=m.predict_proba(x)[:,list(m.classes_).index(1)]; mm=metricas(y,yp); cm=np.array(mm.pop('matriz_confusao')).ravel()
            met.append({'modelo':nome,'cenario':c,**mm,**dict(zip(('vn','fp','fn','vp'),map(int,cm)))})
            mapas[(nome,c)]={r['image_id']:(int(real),int(p)) for r,real,p in zip(rows,y,yp)}
            pred.extend({'image_id':r['image_id'],'cenario':c,'modelo':nome,'classe_verdadeira':int(real),'classe_prevista':int(p),'probabilidade_fake':float(proba),'acerto':int(real==p)} for r,real,p,proba in zip(rows,y,yp,probs))
    if len(met)!=70 or len(pred)!=8400: raise ValueError('Contagens invalidas')
    escrever(SAIDA/'metricas.csv',met); escrever(SAIDA/'predicoes.csv',pred)
    resumo=[]; variacoes=[]
    for nome in modelos:
        grupo=[x for x in met if x['modelo']==nome]; resumo.append({'modelo':nome,**resumir(grupo)})
        original=next(r for r in grupo if r['cenario']=='original')
        for c in CENARIOS[1:]:
            m=next(x for x in grupo if x['cenario']==c); variacoes.append({'modelo':nome,'cenario':c,**{'variacao_'+k:m[k]-original[k] for k in ('acuracia','precisao','recall','f1','especificidade')},**parear(mapas[(nome,'original')],mapas[(nome,c)])})
    escrever(SAIDA/'resumo_modelos.csv',resumo); escrever(SAIDA/'variacoes_metricas.csv',variacoes)
    ab=ablacao(resumo,met,mapas); escrever(SAIDA/'ablacao.csv',ab); figuras(met,resumo,ab)
    for info in protocolo['modelos'].values():
        if sha(info['arquivo'])!=info['sha256']: raise ValueError('Modelo alterado durante avaliacao')
    if sha(CAMINHO)!=json.loads((BASE/'relatorio_finalizacao.json').read_text(encoding='utf-8'))['hash_manifesto'] or sha(MANIFESTO)!=protocolo['manifesto_anterior_sha256']: raise ValueError('Manifesto alterado')
    rel={'manifesto_independente_sha256':sha(CAMINHO),'manifesto_anterior_sha256':sha(MANIFESTO),'protocolo_sha256':sha(BASE/'protocolo_avaliacao.json'),'modelos_sha256':{k:sha(v['arquivo']) for k,v in protocolo['modelos'].items()},'cenarios':CENARIOS,'transformacoes':PARAMETROS,'imagens':120,'instancias':len(caracteristicas),'modelos':10,'combinacoes':len(met),'predicoes':len(pred),'tempo_extracao_segundos':tempo_extracao,'tempo_total_segundos':time.perf_counter()-inicio,'versoes':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__,'scipy':scipy.__version__,'pillow':PIL.__version__},'interpretacao':'Imagens novas do mesmo CIFAKE, transformacoes conhecidas, comparacoes pareadas.'}
    (SAIDA/'relatorio.json').write_text(json.dumps(rel,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print('Concluido:',rel['tempo_total_segundos'],'s')
def ablacao(resumo,met,mapas):
    pares=[]
    for base in ('espacial','espectral','hibrido'):
        pares.append(('aumento',base+'_original',base+'_aumentado'))
    for base in ('espectral','hibrido'):
        for regime in ('original','aumentado'):
            pares.append(('aumento',base+'_original_ampliado',base+'_aumentado_ampliado')) if regime=='original' else None
            pares.append(('ampliacao_espectral',base+'_'+regime,base+'_'+regime+'_ampliado'))
    pares.extend([('abordagem','espacial_aumentado','hibrido_aumentado'),('abordagem','espacial_aumentado','hibrido_aumentado_ampliado'),('abordagem','hibrido_aumentado','hibrido_aumentado_ampliado')])
    out=[]
    for efeito,a,b in pares:
        aa=next(r for r in resumo if r['modelo']==a); bb=next(r for r in resumo if r['modelo']==b)
        for c in CENARIOS:
            x,y=mapas[(a,c)],mapas[(b,c)]
            if set(x)!=set(y) or any(x[i][0]!=y[i][0] for i in x):raise ValueError('Pares divergentes')
            ma=next(r for r in met if r['modelo']==a and r['cenario']==c);mb=next(r for r in met if r['modelo']==b and r['cenario']==c)
            out.append({'efeito':efeito,'referencia':a,'comparado':b,'cenario':c,'delta_f1':mb['f1']-ma['f1'],'delta_f1_original':bb['f1_original']-aa['f1_original'],'delta_f1_medio_transformado':bb['f1_medio_transformados']-aa['f1_medio_transformados'],'delta_pior_f1':bb['pior_f1']-aa['pior_f1'],'acerto_para_erro':sum(x[i][0]==x[i][1] and y[i][0]!=y[i][1] for i in x),'erro_para_acerto':sum(x[i][0]!=x[i][1] and y[i][0]==y[i][1] for i in x),'taxa_mudanca':sum(x[i][1]!=y[i][1] for i in x)/len(x)})
    return out

def figuras(met,resumo,ab):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    def salvar(fig,n): fig.tight_layout();fig.savefig(SAIDA/'figuras'/(n+'.png'),dpi=140);plt.close(fig)
    nomes=[r['modelo'] for r in resumo]
    fig,axes=plt.subplots(2,1,figsize=(11,9),sharey=True)
    for ax,grupo in zip(axes,(nomes[:5],nomes[5:])):
        for n in grupo:ax.plot(CENARIOS,[next(r['f1'] for r in met if r['modelo']==n and r['cenario']==c) for c in CENARIOS],marker='o',label=n)
        ax.set(ylim=(0,1),ylabel='F1');ax.legend(fontsize=8);ax.tick_params(axis='x',rotation=20)
    fig.suptitle('F1 na amostra independente');salvar(fig,'f1_cenarios')
    for campo,n in [('f1_medio_transformados','f1_medio'),('pior_f1','pior_f1'),('maior_queda_f1','maior_queda')]:
        fig,ax=plt.subplots(figsize=(9,5)); ax.barh(nomes,[r[campo] for r in resumo]);ax.set(xlim=(0,1),title=n.replace('_',' '));salvar(fig,n)
    for efeito,nome in [('aumento','comparacao_aumento'),('ampliacao_espectral','comparacao_espectral'),('abordagem','comparacao_abordagens')]:
        grupo=[r for r in ab if r['efeito']==efeito and r['cenario']=='original'];fig,ax=plt.subplots(figsize=(10,4));ax.barh([r['comparado']+' vs '+r['referencia'] for r in grupo],[r['delta_f1_medio_transformado'] for r in grupo]);ax.axvline(0,color='black');ax.set(xlim=(-1,1),title='Diferenca no F1 medio transformado');salvar(fig,nome)
    for nome in nomes:
        fig,axes=plt.subplots(1,7,figsize=(14,3))
        for ax,c in zip(axes,CENARIOS):
            r=next(r for r in met if r['modelo']==nome and r['cenario']==c);mat=np.array([[r['vn'],r['fp']],[r['fn'],r['vp']]]);ax.imshow(mat,vmin=0,vmax=60,cmap='Blues');ax.set(title=c,xticks=[0,1],yticks=[0,1],xticklabels=['R','F'],yticklabels=['R','F'])
            for (i,j),v in np.ndenumerate(mat):ax.text(j,i,str(v),ha='center',va='center')
        fig.suptitle('Matrizes de confusao - '+nome+' (linhas verdadeiras)');salvar(fig,'matrizes_'+nome)
    fig,axes=plt.subplots(1,2,figsize=(13,8))
    for ax,grupo in zip(axes,(nomes[:5],nomes[5:])):
        values=np.array([[next(r['f1'] for r in met if r['modelo']==n and r['cenario']==c) for c in CENARIOS] for n in grupo]);im=ax.imshow(values,vmin=0,vmax=1,cmap='Blues',aspect='auto');ax.set(yticks=range(len(grupo)),yticklabels=grupo,xticks=range(7),xticklabels=CENARIOS,title='F1 por cenario');ax.tick_params(axis='x',rotation=45)
    fig.colorbar(im,ax=axes.ravel().tolist());fig.savefig(SAIDA/'figuras'/'mapa_f1.png',dpi=140,bbox_inches='tight');plt.close(fig)
    grupo=[r for r in ab if r['cenario']=='original'];fig,ax=plt.subplots(figsize=(12,7));ax.barh([r['comparado']+' vs '+r['referencia'] for r in grupo],[r['delta_f1_medio_transformado'] for r in grupo]);ax.axvline(0,color='black');ax.set(xlim=(-1,1),title='Ablacao - diferenca no F1 medio transformado');salvar(fig,'ablacao')
if __name__=='__main__': main()
