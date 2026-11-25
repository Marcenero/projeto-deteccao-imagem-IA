import csv,hashlib,json,unittest
from pathlib import Path
import numpy as np
import joblib
from melhorias.aumento_dados import gerar
from melhorias.treinamento_robusto import resumir,escolher
from experimentos.transformacoes import CENARIOS
from treinamento.modelos import NOMES

class AumentoTests(unittest.TestCase):
    def test_geracao_preserva_origem(self):
        p=Path('data/preparados/cifake/manifest.csv')
        with p.open(encoding='utf-8',newline='') as f: r=next(r for r in csv.DictReader(f) if r['experimental_split']=='train')
        a=gerar([r],p.parent); b=gerar([r],p.parent)
        self.assertEqual([x['cenario'] for x in a],list(CENARIOS))
        for x,y in zip(a,b):
            self.assertEqual((x['image_id'],x['class'],x['experimental_split']),(r['image_id'],int(r['class']),'train'))
            self.assertEqual(x['vetor'].shape,(43,)); np.testing.assert_array_equal(x['vetor'],y['vetor'])
    def test_resumo(self):
        m=[{'cenario':'original','f1':.8},{'cenario':'blur','f1':.5},{'cenario':'jpeg','f1':.9}]; r=resumir(m)
        self.assertAlmostEqual(r['f1_medio_transformados'],.7); self.assertAlmostEqual(r['maior_queda_f1'],.3); self.assertEqual(r['pior_f1'],.5)
    def test_arquivos_experimento(self):
        root=Path('resultados/melhorias/aumento_dados')
        if not (root/'relatorio.json').exists(): self.skipTest('Experimento ainda nao executado')
        with (root/'caracteristicas/treino.csv').open(encoding='utf-8',newline='') as f: tr=list(csv.DictReader(f))
        with (root/'caracteristicas/validacao.csv').open(encoding='utf-8',newline='') as f: va=list(csv.DictReader(f))
        self.assertEqual((len(tr),len(va)),(1470,210)); self.assertEqual({r['experimental_split'] for r in tr},{'train'}); self.assertEqual({r['experimental_split'] for r in va},{'validation'})
        self.assertTrue({r['image_id'] for r in tr}.isdisjoint({r['image_id'] for r in va}))
        with open('data/preparados/cifake/manifest.csv',encoding='utf-8',newline='') as f: test={r['image_id'] for r in csv.DictReader(f) if r['experimental_split']=='test'}
        self.assertTrue(test.isdisjoint({r['image_id'] for r in tr+va}))
        with (root/'predicoes_validacao.csv').open(encoding='utf-8',newline='') as f: self.assertEqual(len(list(csv.DictReader(f))),1260)
        with (root/'metricas_validacao.csv').open(encoding='utf-8',newline='') as f: self.assertEqual(len(list(csv.DictReader(f))),42)
        old=json.loads(Path('resultados/treinamento/relatorio.json').read_text(encoding='utf-8'))
        for tipo in NOMES:
            self.assertEqual(hashlib.sha256(Path(old['modelos'][tipo]['arquivo']).read_bytes()).hexdigest(),old['modelos'][tipo]['sha256'])
            m=joblib.load(root/'modelos'/f'modelo_{tipo}_aumentado.joblib'); self.assertEqual(m.n_features_in_,len(NOMES[tipo])); self.assertEqual(escolher(np.zeros((2,43)),tipo).shape[1],len(NOMES[tipo]))
if __name__=='__main__': unittest.main()
