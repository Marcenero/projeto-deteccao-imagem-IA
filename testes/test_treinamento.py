import unittest
import tempfile
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from treinamento.modelos import carregar, NOMES
from treinamento.avaliar import metricas

class TreinamentoTests(unittest.TestCase):
    def test_dados(self):
        d=carregar()
        for k,v in d.items():
            self.assertEqual(v['X'].shape,(240,len(NOMES[k])))
            self.assertEqual(sum(v['splits']=='train'),210)
            self.assertEqual(sum(v['splits']=='validation'),30)
            self.assertNotIn('test',v['splits'])
            self.assertTrue(set(NOMES[k]).isdisjoint({'image_id','class','experimental_split'}))
    def test_colunas_invalidas(self):
        with tempfile.TemporaryDirectory() as tmp:
            from shutil import copytree
            from pathlib import Path
            copytree('resultados/caracteristicas',Path(tmp)/'dados')
            p=Path(tmp)/'dados/caracteristicas_espectrais.csv'; s=p.read_text(encoding='utf-8'); p.write_text(s.replace('E1,E2,E3','E2,E1,E3',1),encoding='utf-8')
            with self.assertRaises(ValueError): carregar(Path(tmp)/'dados')
    def test_metricas_e_zero(self):
        m=metricas([0,0,1,1],[0,1,0,1]); self.assertEqual(m['matriz_confusao'],[[1,1],[1,1]])
        self.assertEqual(m['precisao'],.5); self.assertEqual(m['especificidade'],.5)
        m=metricas([0,0],[0,0]); self.assertEqual(m['recall'],0); self.assertEqual(m['precisao'],0)
        self.assertEqual(metricas([1],[1])['especificidade'],0)
    def test_reproducao_persistencia(self):
        d=carregar()['espectral']; tr=d['splits']=='train'; va=~tr
        kwargs=dict(n_estimators=10,random_state=42,n_jobs=1)
        a=RandomForestClassifier(**kwargs).fit(d['X'][tr],d['y'][tr]); b=RandomForestClassifier(**kwargs).fit(d['X'][tr],d['y'][tr])
        np.testing.assert_array_equal(a.predict(d['X'][va]),b.predict(d['X'][va]))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'modelo.joblib'; joblib.dump(a,p); np.testing.assert_array_equal(a.predict_proba(d['X'][va]),joblib.load(p).predict_proba(d['X'][va]))
if __name__=='__main__': unittest.main()
