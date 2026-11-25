"""Transformacoes independentes sobre imagens RGB."""
from io import BytesIO
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
CENARIOS=('original','jpeg_75','jpeg_30','resize_75','resize_50','blur_05','blur_10')
PARAMETROS={'jpeg':{'quality':[75,30],'subsampling':0,'optimize':False,'progressive':False},'resize':{'tamanhos':[24,16],'reducao':'Lanczos','retorno':'bilinear'},'blur':{'sigma':[0.5,1.0],'mode':'reflect','truncate':4.0,'conversao':'arredondamento para inteiro mais proximo, uint8'},'entrada':'RGB 32x32'}
def transformar(imagem,cenario):
    if imagem.mode!='RGB' or imagem.size!=(32,32): raise ValueError('Esperado RGB 32x32')
    if cenario=='original': return imagem.copy()
    if cenario.startswith('jpeg_'):
        b=BytesIO(); imagem.save(b,format='JPEG',quality=int(cenario.split('_')[1]),subsampling=0,optimize=False,progressive=False); b.seek(0)
        with Image.open(b) as i: return i.convert('RGB')
    if cenario in ('resize_75','resize_50'):
        n=24 if cenario=='resize_75' else 16
        return imagem.resize((n,n),Image.Resampling.LANCZOS).resize((32,32),Image.Resampling.BILINEAR)
    if cenario in ('blur_05','blur_10'):
        sigma=.5 if cenario=='blur_05' else 1.0
        a=gaussian_filter(np.asarray(imagem,dtype=np.float64),sigma=(sigma,sigma,0),mode='reflect',truncate=4.0)
        return Image.fromarray(np.rint(np.clip(a,0,255)).astype(np.uint8))
    raise ValueError('Cenario desconhecido')
