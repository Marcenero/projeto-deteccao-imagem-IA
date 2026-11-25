import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from extracao_caracteristicas import extrair, NOMES_ESPACIAIS, NOMES_ESPECTRAIS, NOMES_HIBRIDOS
from extracao_caracteristicas.espaciais import extrair as espacial, cinza
from extracao_caracteristicas.extrair import executar, METADADOS
from analise_frequencias.espectro import spectrum

class CaracteristicasTests(unittest.TestCase):
    def test_constante_histogramas_e_dimensoes(self):
        a=np.full((32,32,3), .5); r=extrair(a); v=dict(zip(NOMES_ESPACIAIS,r['espaciais']))
        self.assertEqual([len(r[k]) for k in ('espaciais','espectrais','hibridas')],[40,3,43])
        self.assertTrue(np.isfinite(r['hibridas']).all())
        self.assertTrue(np.all(r['espectrais']==0))
        self.assertEqual(v['corr_rg'],0); self.assertEqual(v['variancia_laplaciano'],0)
        for c in 'rgb':
            self.assertEqual(sum(v[f'hist_{c}_{i}'] for i in range(8)),1)
            self.assertEqual(v[f'hist_{c}_4'],1)
        self.assertEqual(NOMES_HIBRIDOS, NOMES_ESPACIAIS+NOMES_ESPECTRAIS)
    def test_rgb_correlacao_diferencas_laplaciano(self):
        a=np.zeros((32,32,3)); a[:,16:,0]=1; a[:,:,1]=a[:,:,0]; a[:,:,2]=1-a[:,:,0]
        v=dict(zip(NOMES_ESPACIAIS,espacial(a)))
        self.assertAlmostEqual(v['media_r'],.5); self.assertAlmostEqual(v['desvio_r'],.5)
        self.assertAlmostEqual(v['corr_rg'],1); self.assertAlmostEqual(v['corr_rb'],-1)
        self.assertAlmostEqual(v['dif_horizontal_r'],1/31); self.assertEqual(v['dif_vertical_r'],0)
        self.assertAlmostEqual(v['variancia_laplaciano'],2*(.299+.587-.114)**2/32)
        self.assertTrue(np.array_equal(espacial(a),espacial(a)))
    def test_espectral_e_pillow(self):
        a=np.random.default_rng(3).integers(0,256,(32,32,3),dtype=np.uint8)
        resultado=extrair(Image.fromarray(a)); referencia=spectrum(cinza(a.astype(float)/255))['features']
        np.testing.assert_allclose(resultado['espectrais'],[referencia[k] for k in NOMES_ESPECTRAIS],rtol=0,atol=0)
        np.testing.assert_array_equal(resultado['hibridas'],extrair(a)['hibridas'])
    def test_rejeita_valores_invalidos(self):
        a=np.zeros((32,32,3)); a[0,0,0]=np.nan
        with self.assertRaises(ValueError): extrair(a)
    def test_arquivos_reprodutiveis_e_particoes(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            executar(saida=a); executar(saida=b)
            tabelas=[]
            for tipo,nomes in [('espaciais',NOMES_ESPACIAIS),('espectrais',NOMES_ESPECTRAIS),('hibridas',NOMES_HIBRIDOS)]:
                p=Path(a)/f'caracteristicas_{tipo}.csv'; q=Path(b)/p.name
                self.assertEqual(p.read_bytes(),q.read_bytes())
                with p.open(encoding='utf-8',newline='') as f:
                    rd=csv.DictReader(f); self.assertEqual(rd.fieldnames,list(METADADOS+nomes)); rows=list(rd)
                self.assertEqual(len(rows),240); self.assertEqual({x['experimental_split'] for x in rows},{'train','validation'})
                self.assertEqual(len({x['image_id'] for x in rows}),240)
                self.assertTrue(np.isfinite(np.array([[float(x[n]) for n in nomes] for x in rows])).all())
                tabelas.append([[r[k] for k in METADADOS] for r in rows])
            self.assertEqual(tabelas[0],tabelas[1]); self.assertEqual(tabelas[1],tabelas[2])

if __name__=='__main__': unittest.main()
