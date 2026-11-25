import csv,json,unittest
from collections import Counter
from pathlib import Path
import numpy as np
from preparacao_independente.finalizar import DESTINO,escolher
from preparacao_independente.preparar import BASE,ler,sha
from melhorias.avaliacao_independente import validar_entrada,nomes_modelo
from treinamento.avaliar import metricas
from experimentos.robustez import parear

class AvaliacaoIndependenteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=ler(DESTINO/'manifest.csv'); cls.pred=ler(BASE/'avaliacao/predicoes.csv'); cls.met=ler(BASE/'avaliacao/metricas.csv')
    def test_manifesto(self):
        self.assertEqual(Counter(r['class'] for r in self.rows),{'0':60,'1':60}); self.assertEqual(len({r['sha256'] for r in self.rows}),120)
        escolhidos=escolher(ler(BASE/'candidatos.csv'),43); self.assertEqual([r['image_id'] for r in self.rows],[r['image_id'] for r in escolhidos]); self.assertEqual(escolhidos,escolher(ler(BASE/'candidatos.csv'),43))
        antigo=ler(Path('data/preparados/cifake/manifest.csv'))+ler(Path('resultados/cifake/selecao_categoria.csv')); self.assertTrue({r['image_id'] for r in self.rows}.isdisjoint({r['image_id'] for r in antigo})); self.assertTrue({r['sha256'] for r in self.rows}.isdisjoint({r['sha256'] for r in antigo}))
        self.assertTrue(all(sha(DESTINO/r['local_path'])==r['sha256'] for r in self.rows))
    def test_modelos_e_entradas(self):
        protocolo,rows,modelos=validar_entrada(); self.assertEqual(len(modelos),10); self.assertEqual(len(rows),120)
        for nome,m in modelos.items(): self.assertEqual(m.n_features_in_,len(nomes_modelo(nome)))
    def test_predicoes_e_metricas(self):
        self.assertEqual(len(self.pred),8400); self.assertEqual(len(self.met),70)
        grupos=Counter((r['modelo'],r['cenario']) for r in self.pred); self.assertEqual(len(grupos),70); self.assertEqual(set(grupos.values()),{120}); self.assertTrue(all(np.isfinite(float(r['probabilidade_fake'])) for r in self.pred))
        for m in self.met:
            p=[r for r in self.pred if r['modelo']==m['modelo'] and r['cenario']==m['cenario']]; z=metricas([int(r['classe_verdadeira']) for r in p],[int(r['classe_prevista']) for r in p])
            for k in ('acuracia','precisao','recall','f1','especificidade'): self.assertAlmostEqual(z[k],float(m[k]))
    def test_ablacao_pareada(self):
        ab=ler(BASE/'avaliacao/ablacao.csv')
        for a in ab:
            def mapa(nome):return {r['image_id']:(int(r['classe_verdadeira']),int(r['classe_prevista'])) for r in self.pred if r['modelo']==nome and r['cenario']==a['cenario']}
            x,y=mapa(a['referencia']),mapa(a['comparado']); v=parear(x,y);self.assertEqual(v['acerto_para_erro'],int(a['acerto_para_erro']));self.assertEqual(v['erro_para_acerto'],int(a['erro_para_acerto']))
            f=lambda nome:float(next(r['f1'] for r in self.met if r['modelo']==nome and r['cenario']==a['cenario']))
            self.assertAlmostEqual(float(a['delta_f1']),f(a['comparado'])-f(a['referencia']))
if __name__=='__main__':unittest.main()
