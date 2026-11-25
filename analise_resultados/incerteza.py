"""Bootstrap estratificado e pareado por imagem."""
import numpy as np

def indices_estratificados(y,n=10000,seed=42):
    y=np.asarray(y); g=np.random.default_rng(seed); grupos=[np.flatnonzero(y==k) for k in (0,1)]
    if any(len(x)==0 for x in grupos) or sum(map(len,grupos))!=len(y): raise ValueError('Classes invalidas')
    return np.concatenate([g.choice(x,size=(n,len(x)),replace=True) for x in grupos],axis=1)

def metricas_amostras(y,p,indices):
    yt=np.asarray(y)[indices]; yp=np.asarray(p)[indices]
    tp=((yt==1)&(yp==1)).sum(axis=1); fp=((yt==0)&(yp==1)).sum(axis=1); fn=((yt==1)&(yp==0)).sum(axis=1)
    den=2*tp+fp+fn; f1=np.divide(2*tp,den,out=np.zeros(len(den),dtype=float),where=den!=0)
    return f1,(yt==yp).mean(axis=1)

def intervalo(a): return tuple(float(x) for x in np.quantile(a,[.025,.975]))
