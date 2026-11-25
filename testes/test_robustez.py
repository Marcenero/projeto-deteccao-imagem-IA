import unittest
import numpy as np
from PIL import Image
from experimentos.transformacoes import transformar,CENARIOS
from experimentos.robustez import variacao,parear
from extracao_caracteristicas import espaciais, espectrais

class RobustezTests(unittest.TestCase):
    def test_transformacoes(self):
        a=np.random.default_rng(7).integers(0,256,(32,32,3),dtype=np.uint8); img=Image.fromarray(a)
        self.assertEqual(len(CENARIOS),7)
        for c in CENARIOS:
            x=transformar(img,c); self.assertEqual(x.size,(32,32)); self.assertEqual(x.mode,'RGB'); self.assertTrue(np.array_equal(a,np.array(img)))
            np.testing.assert_array_equal(np.array(x),np.array(transformar(img,c)))
            v=np.concatenate((espaciais.extrair(x),espectrais.extrair(x))); self.assertEqual(len(v),43); self.assertTrue(np.isfinite(v).all())
    def test_pareamento(self):
        b={'a':(0,0),'b':(1,0),'c':(1,1)}; a={'a':(0,1),'b':(1,1),'c':(1,1)}
        self.assertEqual(parear(b,a),{'acerto_para_erro':1,'erro_para_acerto':1,'taxa_mudanca':2/3})
        with self.assertRaises(ValueError): parear(b,{'a':(0,0)})
    def test_resultados_existentes(self):
        import csv
        from collections import Counter
        from pathlib import Path
        pasta=Path('resultados/robustez')
        if not (pasta/'predicoes.csv').exists(): self.skipTest('Experimento ainda nao executado')
        with (pasta/'predicoes.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),1260)
        self.assertEqual(set(Counter((r['modelo'],r['cenario']) for r in rows).values()),{60})
        self.assertEqual(len({r['image_id'] for r in rows}),60)
        for r in rows: self.assertTrue(np.isfinite(float(r['probabilidade_fake'])))
        with (pasta/'metricas.csv').open(encoding='utf-8',newline='') as f: self.assertEqual(len(list(csv.DictReader(f))),21)
    def test_variacoes(self):
        b=dict.fromkeys(('acuracia','precisao','recall','f1','especificidade'),.5); a=dict.fromkeys(b,.75)
        self.assertTrue(all(x==.25 for x in variacao(b,a).values()))
if __name__=='__main__': unittest.main()
