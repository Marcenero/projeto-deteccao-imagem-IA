import unittest
from pathlib import Path
import json,hashlib
from preparacao_independente.preparar import selecionar,SEED

class IndependenteTests(unittest.TestCase):
    def test_sorteio_balanceado_reprodutivel(self):
        rows=[{'class':cl,'image_id':f'test/{cl}/{i}'} for cl in ('0','1') for i in range(20)]
        a=selecionar(rows,5); self.assertEqual(a,selecionar(list(reversed(rows)),5)); self.assertEqual(len(a),10); self.assertEqual(sum(r['class']=='0' for r in a),5)
        with self.assertRaises(ValueError): selecionar(rows,21)
    def test_integridade_protocolo(self):
        p=Path('resultados/melhorias/teste_independente/protocolo_avaliacao.json'); r=json.loads(p.read_text(encoding='utf-8')); h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        self.assertEqual(len(r['modelos']),10); self.assertEqual(h('data/preparados/cifake/manifest.csv'),r['manifesto_anterior_sha256'])
        for m in r['modelos'].values():self.assertEqual(h(m['arquivo']),m['sha256'])
    def test_candidatos_se_existirem(self):
        import csv
        p=Path('resultados/melhorias/teste_independente/candidatos.csv')
        if not p.exists():self.skipTest('Preparacao ainda nao concluida')
        with p.open(encoding='utf-8',newline='') as f:r=list(csv.DictReader(f))
        self.assertEqual(len(r),2000); self.assertTrue(all(x['original_split']=='test' for x in r)); self.assertEqual(len({x['sha256'] for x in r}),2000)
        with open('resultados/cifake/selecao_categoria.csv',encoding='utf-8',newline='') as f:hist=list(csv.DictReader(f))
        self.assertTrue({x['image_id'] for x in r}.isdisjoint({x['image_id'] for x in hist}));self.assertTrue({x['sha256'] for x in r}.isdisjoint({x['sha256'] for x in hist}))
    def test_folhas_e_manifesto(self):
        import csv
        raiz=Path('resultados/melhorias/teste_independente')
        with (raiz/'candidatos.csv').open(encoding='utf-8',newline='') as f: candidatos=list(csv.DictReader(f))
        with (raiz/'mapa_revisao.csv').open(encoding='utf-8',newline='') as f: mapa=list(csv.DictReader(f))
        self.assertEqual(len({r['image_id'] for r in mapa}),len(mapa))
        ids={r['image_id'] for r in candidatos}; self.assertTrue({r['image_id'] for r in mapa}.issubset(ids))
        self.assertEqual(len({(r['sheet'],r['position']) for r in mapa}),len(mapa)); self.assertTrue(all((raiz/'folhas_revisao'/r['sheet']).exists() for r in mapa))
        with open('data/preparados/cifake/manifest.csv',encoding='utf-8',newline='') as f: antigo=list(csv.DictReader(f))
        self.assertTrue({r['image_id'] for r in candidatos}.isdisjoint({r['image_id'] for r in antigo})); self.assertTrue({r['sha256'] for r in candidatos}.isdisjoint({r['sha256'] for r in antigo}))
        self.assertEqual({r['original_split'] for r in candidatos},{'test'})
if __name__=='__main__':unittest.main()
