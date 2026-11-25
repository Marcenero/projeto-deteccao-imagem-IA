import unittest
import numpy as np
from analise_resultados.incerteza import indices_estratificados,metricas_amostras,intervalo
from analise_resultados.consolidar import carregar,bootstrap,analise_erros
from treinamento.avaliar import metricas

class AnaliseResultadosTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.protocolo,cls.manifesto,cls.y,cls.nomes,cls.arr,cls.met,cls.erro=carregar()
    def test_integridade_recalculo(self):
        self.assertEqual(self.arr.shape,(10,7,120));self.assertEqual(self.erro,0);self.assertEqual(len(self.met),70)
        self.assertEqual(sum(self.y==0),60);self.assertEqual(sum(self.y==1),60)
    def test_bootstrap_estratificado(self):
        y=np.array([0]*60+[1]*60); a=indices_estratificados(y,500,42); b=indices_estratificados(y,500,42)
        np.testing.assert_array_equal(a,b);self.assertTrue(np.all((y[a]==0).sum(axis=1)==60));self.assertTrue(np.all((y[a]==1).sum(axis=1)==60))
    def test_f1_e_zero(self):
        y=np.array([0,0,1,1]);p=np.array([0,1,0,1]);idx=np.array([[0,1,2,3],[0,0,0,0]])
        f,a=metricas_amostras(y,p,idx);self.assertEqual(f[0],metricas(y,p)['f1']);self.assertEqual(f[1],0);self.assertTrue(np.isfinite(f).all());self.assertEqual(a[0],.5)
    def test_bootstrap_pareado_reprodutivel(self):
        a,b=bootstrap(self.y,self.nomes,self.arr,self.met,n=50,seed=42); c,d=bootstrap(self.y,self.nomes,self.arr,self.met,n=50,seed=42)
        self.assertEqual(a,c);self.assertEqual(b,d);self.assertTrue(all(np.isfinite(r['ic_95_inferior']) and np.isfinite(r['ic_95_superior']) for r in a+b))
        x=np.arange(12,dtype=float);self.assertEqual(intervalo(x-x),(0.,0.))
    def test_pareamento_erros(self):
        mud,div,ex=analise_erros(self.nomes,self.arr,self.y,[r['image_id'] for r in self.manifesto]);self.assertEqual(len(mud),60);self.assertTrue(all(r['erros_em_ambos']+r['erros_so_referencia']+r['erros_so_comparado']+r['ambos_corretos']==120 for r in div))
if __name__=='__main__':unittest.main()
