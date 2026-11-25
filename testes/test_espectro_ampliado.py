import csv,hashlib,json,unittest
from pathlib import Path
import numpy as np
from melhorias.espectro_ampliado import extrair,entropia,mascaras,NOMES
from analise_frequencias.espectro import spectrum
from extracao_caracteristicas.espaciais import cinza

class EspectroAmpliadoTests(unittest.TestCase):
    def test_cobertura_dc(self):
        f=np.fft.fftshift(np.fft.fftfreq(32)); r=np.hypot(f[:,None],f[None,:]); m=mascaras(r); s=np.sum(m,axis=0)
        np.testing.assert_array_equal(s,((r>0)&(r<=.5)).astype(int)); self.assertFalse(s[16,16]); self.assertEqual(len(m),7)
    def test_sinal_e_energia(self):
        x=np.arange(32); g=np.tile((.5+.4*np.cos(2*np.pi*4*x/32)),(32,1)); img=np.repeat(g[:,:,None],3,axis=2); v,d=extrair(img); prev=spectrum(cinza(img))['features']
        np.testing.assert_allclose(v[:3],[prev[k] for k in ('E1','E2','E3')],atol=1e-14); self.assertAlmostEqual(sum(v[:6])+d['B1'],1); self.assertAlmostEqual(v[6],.125); self.assertEqual(v.shape,(8,)); self.assertTrue(np.isfinite(v).all()); self.assertEqual(NOMES,('E1','E2','E3','B2','B3','B4','centroide','entropia'))
    def test_entropia_constante(self):
        self.assertAlmostEqual(entropia(np.ones(8)/8),1); self.assertEqual(entropia(np.array([1.,0,0,0])),0)
        v,d=extrair(np.full((32,32,3),.5)); self.assertTrue(d['energia_nula']); self.assertTrue(np.all(v==0))
    def test_artefatos(self):
        root=Path('resultados/melhorias/espectro_ampliado'); rel=json.loads((root/'relatorio.json').read_text(encoding='utf-8')); self.assertEqual(rel['contagens']['combinacoes'],70); self.assertEqual(rel['contagens']['novas_predicoes'],840)
        with (root/'caracteristicas/treino.csv').open(encoding='utf-8',newline='') as f: tr=list(csv.DictReader(f))
        with (root/'caracteristicas/validacao.csv').open(encoding='utf-8',newline='') as f: va=list(csv.DictReader(f))
        self.assertEqual((len(tr),len(va)),(1470,210)); self.assertEqual({r['experimental_split'] for r in tr},{'train'}); self.assertEqual({r['experimental_split'] for r in va},{'validation'}); self.assertTrue({r['image_id'] for r in tr}.isdisjoint({r['image_id'] for r in va}))
        with open('data/preparados/cifake/manifest.csv',encoding='utf-8',newline='') as f: teste={r['image_id'] for r in csv.DictReader(f) if r['experimental_split']=='test'}
        self.assertTrue(teste.isdisjoint({r['image_id'] for r in tr+va}))
        orig=json.loads(Path('resultados/treinamento/relatorio.json').read_text(encoding='utf-8')); aug=json.loads(Path('resultados/melhorias/aumento_dados/relatorio.json').read_text(encoding='utf-8'))
        for k in ('espacial','espectral','hibrido'):
            for p,h in ((Path(orig['modelos'][k]['arquivo']),orig['modelos'][k]['sha256']),(Path('resultados/melhorias/aumento_dados/modelos')/f'modelo_{k}_aumentado.joblib',aug['modelos_aumentados'][k])): self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),h)
if __name__=='__main__': unittest.main()
