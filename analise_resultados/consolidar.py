"""Consolida os experimentos sem alterar os artefatos historicos."""
import argparse,csv,hashlib,json,time,shutil
from collections import Counter
from pathlib import Path
import numpy as np
from treinamento.avaliar import metricas
from experimentos.transformacoes import CENARIOS
from experimentos.robustez import parear
from melhorias.treinamento_robusto import escrever,resumir
from .incerteza import indices_estratificados,metricas_amostras,intervalo
from .comparacoes import PARES

BASE=Path('resultados/melhorias/teste_independente'); ENTRADA=BASE/'avaliacao'; SAIDA=Path('resultados/analise_final')
def ler(p):
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def carregar():
    protocolo=json.loads((BASE/'protocolo_avaliacao.json').read_text(encoding='utf-8')); avaliacao=json.loads((ENTRADA/'relatorio.json').read_text(encoding='utf-8')); final=json.loads((BASE/'relatorio_finalizacao.json').read_text(encoding='utf-8'))
    original=Path('data/preparados/cifake/manifest.csv'); independente=Path('data/preparados/cifake_teste_independente/manifest.csv')
    if sha(original)!=protocolo['manifesto_anterior_sha256'] or sha(independente)!=final['hash_manifesto'] or sha(BASE/'protocolo_avaliacao.json')!=avaliacao['protocolo_sha256']:raise ValueError('Integridade de manifestos ou protocolo')
    for k,v in protocolo['modelos'].items():
        if sha(v['arquivo'])!=v['sha256'] or v['sha256']!=avaliacao['modelos_sha256'][k]: raise ValueError('Modelo divergente')
    manifesto=ler(independente); anterior=ler(original)
    if len(manifesto)!=120 or Counter(r['class'] for r in manifesto)!={'0':60,'1':60} or {r['image_id'] for r in manifesto}&{r['image_id'] for r in anterior} or {r['sha256'] for r in manifesto}&{r['sha256'] for r in anterior}: raise ValueError('Separacao ou composicao invalida')
    ids=[r['image_id'] for r in manifesto]; y=np.array([int(r['class']) for r in manifesto]); ind={v:i for i,v in enumerate(ids)}; nomes=list(protocolo['modelos']); preds=ler(ENTRADA/'predicoes.csv'); historico=ler(ENTRADA/'metricas.csv')
    if len(ids)!=len(ind) or len(nomes)!=10 or tuple(protocolo['cenarios'])!=CENARIOS or len(preds)!=8400 or len(historico)!=70:raise ValueError('Contagens invalidas')
    arr=np.full((10,7,120),-1,dtype=int); seen=set()
    for r in preds:
        k=(r['modelo'],r['cenario'],r['image_id'])
        if k in seen or k[0] not in nomes or k[1] not in CENARIOS or k[2] not in ind:raise ValueError('Predicao duplicada ou desconhecida')
        i=ind[k[2]]; v=int(r['classe_prevista']); prob=float(r['probabilidade_fake'])
        if int(r['classe_verdadeira'])!=y[i] or v not in (0,1) or not np.isfinite(prob) or not 0<=prob<=1 or int(r['acerto'])!=int(v==y[i]):raise ValueError('Predicao invalida')
        arr[nomes.index(k[0]),CENARIOS.index(k[1]),i]=v;seen.add(k)
    if (arr<0).any():raise ValueError('Predicoes incompletas')
    met=[]; diferenca=0.
    for i,n in enumerate(nomes):
        for j,c in enumerate(CENARIOS):
            m=metricas(y,arr[i,j]); mat=np.array(m.pop('matriz_confusao')).ravel(); old=[r for r in historico if (r['modelo'],r['cenario'])==(n,c)]
            if len(old)!=1:raise ValueError('Metrica historica ausente ou duplicada')
            for k in m: diferenca=max(diferenca,abs(m[k]-float(old[0][k])))
            if diferenca>1e-12 or list(mat)!=[int(old[0][k]) for k in ('vn','fp','fn','vp')]:raise ValueError('Metricas historicas divergentes')
            met.append({'modelo':n,'cenario':c,**m,**dict(zip(('vn','fp','fn','vp'),map(int,mat)))})
    return protocolo,manifesto,y,nomes,arr,met,diferenca

def bootstrap(y,nomes,arr,met,n=10000,seed=42):
    indices=indices_estratificados(y,n,seed); series={}; rows=[]
    for i,nome in enumerate(nomes):
        fs=[]
        for j,c in enumerate(CENARIOS):
            f,ac=metricas_amostras(y,arr[i,j],indices);fs.append(f)
            estimado=next(r for r in met if r['modelo']==nome and r['cenario']==c)
            for metrica,valores in [('f1',f)]+([('acuracia',ac)] if c=='original' else []):
                lo,hi=intervalo(valores);rows.append({'modelo':nome,'cenario':c,'metrica':metrica,'estimativa':estimado[metrica],'ic_95_inferior':lo,'ic_95_superior':hi})
        media=np.mean(fs[1:],axis=0);series[nome]=media;lo,hi=intervalo(media)
        rows.append({'modelo':nome,'cenario':'media_transformados','metrica':'f1','estimativa':np.mean([r['f1'] for r in met if r['modelo']==nome and r['cenario']!='original']),'ic_95_inferior':lo,'ic_95_superior':hi})
    diferencas=[]
    for efeito,a,b in PARES:
        d=series[b]-series[a];lo,hi=intervalo(d)
        valor=np.mean([r['f1'] for r in met if r['modelo']==b and r['cenario']!='original'])-np.mean([r['f1'] for r in met if r['modelo']==a and r['cenario']!='original'])
        diferencas.append({'efeito':efeito,'referencia':a,'comparado':b,'diferenca_f1_medio':valor,'diferenca_pontos_percentuais':valor*100,'ic_95_inferior':lo,'ic_95_superior':hi})
    return rows,diferencas

def analise_erros(nomes,arr,y,ids):
    mudancas=[]; divergencias=[]; exemplos=[]
    for i,n in enumerate(nomes):
        base={k:(int(y[t]),int(arr[i,0,t])) for t,k in enumerate(ids)}
        for j,c in enumerate(CENARIOS[1:],1):
            atual={k:(int(y[t]),int(arr[i,j,t])) for t,k in enumerate(ids)};mudancas.append({'modelo':n,'cenario':c,**parear(base,atual)})
    for efeito,a,b in PARES:
        for j,c in enumerate(CENARIOS):
            pa,pb=arr[nomes.index(a),j],arr[nomes.index(b),j]; ea=pa!=y;eb=pb!=y
            divergencias.append({'efeito':efeito,'referencia':a,'comparado':b,'cenario':c,'erros_em_ambos':int((ea&eb).sum()),'erros_so_referencia':int((ea&~eb).sum()),'erros_so_comparado':int((~ea&eb).sum()),'ambos_corretos':int((~ea&~eb).sum()),'previsoes_diferentes':int((pa!=pb).sum())})
            if efeito=='abordagem' and c=='original':
                for t in np.flatnonzero(ea!=eb):exemplos.append({'criterio':'divergencia_acerto_original','referencia':a,'comparado':b,'image_id':ids[t],'classe':int(y[t]),'previsao_referencia':int(pa[t]),'previsao_comparado':int(pb[t])})
    return mudancas,divergencias,exemplos
def espectro():
    dados=ler(ENTRADA/'caracteristicas.csv'); nomes=('B1','B2','B3','B4','E1','E2','E3','centroide','entropia'); out=[]
    if len(dados)!=840: raise ValueError('Caracteristicas espectrais incompletas')
    for c in CENARIOS:
        for cl in ('0','1'):
            g=[r for r in dados if r['cenario']==c and r['class']==cl]
            if len(g)!=60:raise ValueError('Caracteristicas espectrais desbalanceadas')
            for nome in nomes:
                a=np.array([float(r[nome]) for r in g]);out.append({'cenario':c,'class':cl,'caracteristica':nome,'n':len(a),'media':float(a.mean()),'desvio':float(a.std(ddof=1)),'mediana':float(np.median(a)),'q1':float(np.quantile(a,.25)),'q3':float(np.quantile(a,.75))})
    return out

def desenhar(dest,met,resumo,intervalos,difs,espectral):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    def salvar(fig,n): fig.tight_layout();fig.savefig(dest/(n+'.png'),dpi=160);plt.close(fig)
    labels=[r['modelo'] for r in resumo]; fig,ax=plt.subplots(figsize=(10,5));ax.barh(labels,[r['f1_medio_transformados'] for r in resumo]);ax.set(xlim=(0,1),title='F1 medio nos seis cenarios transformados',xlabel='F1');salvar(fig,'06_f1_medio')
    for efeito,n in [('aumento','07_aumento'),('ampliacao_espectral','08_ampliacao')]:
        subset=[r for r in difs if r['efeito']==efeito];fig,ax=plt.subplots(figsize=(10,4));ax.barh([r['comparado']+' - '+r['referencia'] for r in subset],[r['diferenca_pontos_percentuais'] for r in subset]);ax.axvline(0,color='black');ax.set(xlim=(-100,100),xlabel='Diferenca em pontos percentuais',title='Efeito sobre F1 medio transformado');salvar(fig,n)
    principais=('espacial_aumentado','hibrido_aumentado','hibrido_aumentado_ampliado','espectral_aumentado_ampliado');fig,ax=plt.subplots(figsize=(9,4))
    for nome in principais: ax.plot(CENARIOS,[next(r['f1'] for r in met if r['modelo']==nome and r['cenario']==c) for c in CENARIOS],marker='o',label=nome)
    ax.set(ylim=(0,1),title='F1 por cenario: modelos selecionados',ylabel='F1');ax.tick_params(axis='x',rotation=25);ax.legend(fontsize=8);salvar(fig,'09_f1_cenarios')
    rs=[r for r in intervalos if r['cenario']=='media_transformados']; fig,ax=plt.subplots(figsize=(10,5));ax.errorbar([r['estimativa'] for r in rs],range(len(rs)),xerr=[[r['estimativa']-r['ic_95_inferior'] for r in rs],[r['ic_95_superior']-r['estimativa'] for r in rs]],fmt='o',capsize=3);ax.set(yticks=range(len(rs)),yticklabels=[r['modelo'] for r in rs],xlim=(0,1),title='F1 medio transformado: IC bootstrap 95%');salvar(fig,'10_intervalos')
    fig,ax=plt.subplots(figsize=(11,7));x=[r['diferenca_f1_medio'] for r in difs];ax.errorbar(x,range(len(x)),xerr=[[v-r['ic_95_inferior'] for v,r in zip(x,difs)],[r['ic_95_superior']-v for v,r in zip(x,difs)]],fmt='o',capsize=3);ax.axvline(0,color='black');ax.set(yticks=range(len(x)),yticklabels=[r['comparado']+' - '+r['referencia'] for r in difs],title='Diferencas pareadas: IC bootstrap 95%',xlabel='Diferenca no F1 medio');salvar(fig,'11_diferencas_pareadas')
    fig,ax=plt.subplots(figsize=(9,5));
    for n in ('E1','E2','E3','B2','B3','B4','centroide','entropia'):
        ax.plot(CENARIOS,[np.mean([r['media'] for r in espectral if r['cenario']==c and r['caracteristica']==n]) for c in CENARIOS],marker='.',label=n)
    ax.set(ylim=(0,1),title='Caracteristicas espectrais medias: amostra independente');ax.tick_params(axis='x',rotation=25);ax.legend(ncol=2);salvar(fig,'05_efeito_energias')
    fig,ax=plt.subplots(figsize=(12,2));ax.axis('off');ax.text(.5,.7,'CIFAKE automoveis 32 x 32',ha='center',fontsize=14);ax.text(.5,.35,'Treino 210  →  caracteristicas e Random Forest  →  validacao 30  →  avaliacao independente 120',ha='center',fontsize=10);salvar(fig,'01_fluxo_metodologico')

def main():
    global SAIDA
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--saida',type=Path,default=SAIDA);p.add_argument('--reamostragens',type=int,default=10000);args=p.parse_args();SAIDA=args.saida
    if args.reamostragens<=0:raise ValueError('Quantidade de reamostragens invalida')
    if SAIDA.exists():raise FileExistsError('Saida existente; escolha outra pasta')
    t=time.perf_counter(); protocolo,manifesto,y,nomes,arr,met,erro=carregar();SAIDA.mkdir(parents=True);tabelas=SAIDA/'tabelas';fig=SAIDA/'figuras';tabelas.mkdir();fig.mkdir()
    resumo=[{'modelo':n,**resumir([r for r in met if r['modelo']==n])} for n in nomes]; escrever(tabelas/'metricas_independentes.csv',met);escrever(tabelas/'resumo_independente.csv',resumo)
    variacoes=[]
    for n in nomes:
        base=next(r for r in met if r['modelo']==n and r['cenario']=='original')
        for c in CENARIOS[1:]:
            v=next(r for r in met if r['modelo']==n and r['cenario']==c);variacoes.append({'modelo':n,'cenario':c,**{'variacao_'+k:v[k]-base[k] for k in ('acuracia','precisao','recall','f1','especificidade')}})
    escrever(tabelas/'variacoes_metricas.csv',variacoes)
    ints,difs=bootstrap(y,nomes,arr,met,n=args.reamostragens);escrever(tabelas/'intervalos_bootstrap.csv',ints);escrever(tabelas/'diferencas_pareadas_bootstrap.csv',difs)
    ab=[]
    for efeito,a,b in PARES:
        x=next(r for r in resumo if r['modelo']==a);z=next(r for r in resumo if r['modelo']==b);ab.append({'efeito':efeito,'referencia':a,'comparado':b,**{'delta_'+k+'_pp':100*(z[k]-x[k]) for k in ('f1_original','f1_medio_transformados','pior_f1')}})
    escrever(tabelas/'ablacao.csv',ab)
    mudancas,div,exemplos=analise_erros(nomes,arr,y,[r['image_id'] for r in manifesto]);escrever(tabelas/'mudancas_classificacao.csv',mudancas);escrever(tabelas/'divergencias_modelos.csv',div)
    if exemplos: escrever(tabelas/'exemplos_divergentes.csv',exemplos)
    esp=espectro();escrever(tabelas/'espectro_independente.csv',esp)
    for origem,n in [(Path('resultados/robustez/metricas.csv'),'metricas_teste_inicial_60.csv'),(Path('resultados/melhorias/espectro_ampliado/metricas_consolidadas.csv'),'metricas_validacao_exploratoria_30.csv'),(Path('resultados/frequencias/estatisticas_radiais.csv'),'perfis_radiais_treino.csv'),(Path('resultados/melhorias/espectro_ampliado/estatisticas_treino.csv'),'espectro_treino.csv'),(Path('resultados/melhorias/espectro_ampliado/importancia_caracteristicas.csv'),'importancia_modelos_ampliados.csv'),(Path('resultados/treinamento/importancia_caracteristicas.csv'),'importancia_modelos_originais.csv')]: shutil.copyfile(origem,tabelas/n)
    for origem,n in [(Path('resultados/robustez/figuras/exemplos.png'),'02_exemplos_teste_inicial.png'),(Path('resultados/frequencias/figuras/03_perfis_radiais.png'),'03_perfis_radiais_treino.png'),(Path('resultados/melhorias/espectro_ampliado/figuras/faixas.png'),'04_faixas.png'),(ENTRADA/'figuras/matrizes_espacial_aumentado.png','12_matrizes_espacial_aumentado.png'),(ENTRADA/'figuras/matrizes_hibrido_aumentado.png','12_matrizes_hibrido_aumentado.png')]:shutil.copyfile(origem,fig/n)
    desenhar(fig,met,resumo,ints,difs,esp); sintetizar(SAIDA,resumo,difs,div,esp,args.reamostragens)
    rel={'bootstrap':{'reamostragens':args.reamostragens,'seed':42,'estratificacao':'60 REAL e 60 FAKE por imagem original','intervalo':'percentil 95%','condicional':'modelos treinados fixos'},'manifesto_anterior_sha256':sha('data/preparados/cifake/manifest.csv'),'manifesto_independente_sha256':sha('data/preparados/cifake_teste_independente/manifest.csv'),'protocolo_sha256':sha(BASE/'protocolo_avaliacao.json'),'modelos_sha256':{k:sha(v['arquivo']) for k,v in protocolo['modelos'].items()},'entradas_sha256':{n:sha(ENTRADA/n) for n in ('predicoes.csv','metricas.csv','caracteristicas.csv')},'verificacoes':{'predicoes':int(arr.size),'combinacoes':len(met),'diferenca_maxima_metricas':erro,'ids_unicos':len(manifesto),'independencia_hash_id':True},'tempo_segundos':time.perf_counter()-t}
    (SAIDA/'relatorio.json').write_text(json.dumps(rel,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print('Concluido',rel['verificacoes'],'tempo',rel['tempo_segundos'])
def sintetizar(saida,resumo,difs,div,esp,reamostragens=10000):
    linha=lambda n:next(r for r in resumo if r['modelo']==n)
    ic=lambda a,b:next(r for r in difs if r['referencia']==a and r['comparado']==b)
    principais=('espacial_aumentado','hibrido_aumentado','hibrido_aumentado_ampliado')
    a=ic(principais[0],principais[1]); b=ic(principais[0],principais[2]);
    texto=['# Sintese dos resultados','','A avaliacao independente e a evidencia principal. Os resultados da validacao exploratoria (30 imagens) e do teste inicial (60 imagens) permanecem separados da avaliacao independente (120 imagens).','', '## Resultados independentes','', '| Modelo | F1 original | F1 medio transformado | Pior F1 transformado |','| --- | ---: | ---: | ---: |']
    texto += [f"| {r['modelo']} | {r['f1_original']:.4f} | {r['f1_medio_transformados']:.4f} | {r['pior_f1']:.4f} |" for r in resumo]
    texto += ['',f"Hibrido 43 aumentado menos espacial aumentado: {a['diferenca_f1_medio']:.4f}, IC bootstrap 95% [{a['ic_95_inferior']:.4f}, {a['ic_95_superior']:.4f}]. Hibrido 48 aumentado menos espacial aumentado: {b['diferenca_f1_medio']:.4f}, IC [{b['ic_95_inferior']:.4f}, {b['ic_95_superior']:.4f}]. Intervalo contendo zero nao demonstra equivalencia.",'','O modelo espectral original e sensivel a reducao e desfoque; os dados detalhados por cenario e classe estao nas tabelas. O aumento e a ampliacao espectral tiveram efeitos dependentes da representacao; o F1 medio e o pior cenario precisam ser avaliados em conjunto. As medidas espectrais descrevem o que foi observado; elas, correlacoes e importancias por impureza nao demonstram a causa do desempenho. JPEG nao produz degradacao uniforme em todos os modelos e cenarios.','', '## Interpretacao e limitacoes','', 'A validacao foi reutilizada para desenvolver e comparar melhorias, podendo induzir adaptacao a essa amostra. O teste independente compartilha dataset, categoria e transformacoes conhecidas durante o treinamento aumentado; nao demonstra generalizacao para outros geradores ou transformacoes ineditas. A resolucao 32 x 32, o treino pequeno, a validacao de 30 imagens, o dominio de automoveis, a compressao JPEG previa, a construcao do CIFAKE e a selecao semantica assistida limitam a interpretacao. Versoes transformadas de uma mesma imagem sao dependentes. A triagem dHash nao garante ausencia de todas as duplicatas perceptuais.','', f'Os intervalos percentis usam {reamostragens} reamostragens estratificadas por classe e pareadas por imagem. Eles sao condicionais aos modelos treinados, nao incorporam variabilidade de novos treinamentos e as comparacoes multiplas sao exploratorias. Falsos positivos podem levar a contestacoes indevidas de imagens reais; falsos negativos podem permitir que imagens sinteticas sejam aceitas como autenticas. Nao se deve reajustar os modelos a partir desta analise.','', 'Trabalhos futuros incluem outras categorias, fusao tardia e transformacoes ineditas, com dados independentes e protocolo proprio.']
    texto += ['', '## Ablacao e erros observados', '']
    for r in difs:
        if r['efeito'] in ('aumento','ampliacao_espectral'):
            texto.append(f"- {r['efeito']}: {r['comparado']} menos {r['referencia']}: {100*r['diferenca_f1_medio']:.2f} pontos percentuais; IC 95% [{100*r['ic_95_inferior']:.2f}, {100*r['ic_95_superior']:.2f}].")
    pares=[r for r in div if r['efeito']=='abordagem' and r['cenario']=='original']
    for r in pares:
        texto.append(f"- {r['comparado']} versus {r['referencia']}, original: {r['erros_em_ambos']} erros comuns; {r['erros_so_referencia']} so na referencia e {r['erros_so_comparado']} so no comparado.")
    texto += ['', 'No teste independente, o F1 medio transformado do espacial aumentado e dos dois hibridos aumentados ficou proximo. A superioridade numerica do hibrido ampliado sobre o espacial na validacao nao se manteve no teste independente. A ampliacao espectral favoreceu os modelos espectrais, mas nao produziu ganho uniforme nos hibridos. A recompressao JPEG 75 elevou o F1 de alguns modelos, enquanto JPEG 30 reduziu o de outros; nao houve degradacao uniforme.', '', 'As tabelas espectrais mostram B1 a B4, E1 a E3, centroide e entropia por classe e cenario, bem como perfis radiais no treino. A comparacao de medias descreve distribuicoes e nao estabelece mecanismo causal. Os exemplos selecionados sao todas as imagens, em ordem do manifesto, com divergencia de acerto no original nas comparacoes espaciais versus hibridos aumentados.']
    (saida/'sintese_resultados.md').write_text('\n'.join(texto)+'\n',encoding='utf-8')
if __name__=='__main__':main()
