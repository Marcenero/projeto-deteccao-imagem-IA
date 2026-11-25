# Sintese dos resultados

A avaliacao independente e a evidencia principal. Os resultados da validacao exploratoria (30 imagens) e do teste inicial (60 imagens) permanecem separados da avaliacao independente (120 imagens).

## Resultados independentes

| Modelo | F1 original | F1 medio transformado | Pior F1 transformado |
| --- | ---: | ---: | ---: |
| espacial_original | 0.8833 | 0.8370 | 0.7414 |
| espacial_aumentado | 0.8760 | 0.8568 | 0.8182 |
| espectral_original | 0.6667 | 0.2951 | 0.0845 |
| espectral_aumentado | 0.6281 | 0.4572 | 0.3509 |
| hibrido_original | 0.8305 | 0.5669 | 0.3421 |
| hibrido_aumentado | 0.8760 | 0.8557 | 0.8092 |
| espectral_original_ampliado | 0.7200 | 0.4452 | 0.2133 |
| espectral_aumentado_ampliado | 0.7176 | 0.6464 | 0.5882 |
| hibrido_original_ampliado | 0.8205 | 0.6064 | 0.4444 |
| hibrido_aumentado_ampliado | 0.8548 | 0.8508 | 0.8030 |

Hibrido 43 aumentado menos espacial aumentado: -0.0010, IC bootstrap 95% [-0.0068, 0.0038]. Hibrido 48 aumentado menos espacial aumentado: -0.0060, IC [-0.0374, 0.0281]. Intervalo contendo zero nao demonstra equivalencia.

O modelo espectral original e sensivel a reducao e desfoque; os dados detalhados por cenario e classe estao nas tabelas. O aumento e a ampliacao espectral tiveram efeitos dependentes da representacao; o F1 medio e o pior cenario precisam ser avaliados em conjunto. As medidas espectrais descrevem o que foi observado; elas, correlacoes e importancias por impureza nao demonstram a causa do desempenho. JPEG nao produz degradacao uniforme em todos os modelos e cenarios.

## Interpretacao e limitacoes

A validacao foi reutilizada para desenvolver e comparar melhorias, podendo induzir adaptacao a essa amostra. O teste independente compartilha dataset, categoria e transformacoes conhecidas durante o treinamento aumentado; nao demonstra generalizacao para outros geradores ou transformacoes ineditas. A resolucao 32 x 32, o treino pequeno, a validacao de 30 imagens, o dominio de automoveis, a compressao JPEG previa, a construcao do CIFAKE e a selecao semantica assistida limitam a interpretacao. Versoes transformadas de uma mesma imagem sao dependentes. A triagem dHash nao garante ausencia de todas as duplicatas perceptuais.

Os intervalos percentis usam 10000 reamostragens estratificadas por classe e pareadas por imagem. Eles sao condicionais aos modelos treinados, nao incorporam variabilidade de novos treinamentos e as comparacoes multiplas sao exploratorias. Falsos positivos podem levar a contestacoes indevidas de imagens reais; falsos negativos podem permitir que imagens sinteticas sejam aceitas como autenticas. Nao se deve reajustar os modelos a partir desta analise.

Trabalhos futuros incluem outras categorias, fusao tardia e transformacoes ineditas, com dados independentes e protocolo proprio.

## Ablacao e erros observados

- ampliacao_espectral: hibrido_aumentado_ampliado menos hibrido_aumentado: -0.49 pontos percentuais; IC 95% [-3.69, 2.98].
- aumento: espacial_aumentado menos espacial_original: 1.97 pontos percentuais; IC 95% [-2.51, 6.54].
- aumento: espectral_aumentado menos espectral_original: 16.21 pontos percentuais; IC 95% [10.64, 21.73].
- aumento: hibrido_aumentado menos hibrido_original: 28.88 pontos percentuais; IC 95% [20.10, 38.32].
- aumento: espectral_aumentado_ampliado menos espectral_original_ampliado: 20.13 pontos percentuais; IC 95% [12.83, 27.61].
- ampliacao_espectral: espectral_original_ampliado menos espectral_original: 15.01 pontos percentuais; IC 95% [8.05, 22.23].
- ampliacao_espectral: espectral_aumentado_ampliado menos espectral_aumentado: 18.92 pontos percentuais; IC 95% [9.95, 28.15].
- aumento: hibrido_aumentado_ampliado menos hibrido_original_ampliado: 24.44 pontos percentuais; IC 95% [15.68, 33.78].
- ampliacao_espectral: hibrido_original_ampliado menos hibrido_original: 3.95 pontos percentuais; IC 95% [-0.74, 8.81].
- ampliacao_espectral: hibrido_aumentado_ampliado menos hibrido_aumentado: -0.49 pontos percentuais; IC 95% [-3.69, 2.98].
- hibrido_aumentado versus espacial_aumentado, original: 15 erros comuns; 0 so na referencia e 0 so no comparado.
- hibrido_aumentado_ampliado versus espacial_aumentado, original: 13 erros comuns; 2 so na referencia e 5 so no comparado.

No teste independente, o F1 medio transformado do espacial aumentado e dos dois hibridos aumentados ficou proximo. A superioridade numerica do hibrido ampliado sobre o espacial na validacao nao se manteve no teste independente. A ampliacao espectral favoreceu os modelos espectrais, mas nao produziu ganho uniforme nos hibridos. A recompressao JPEG 75 elevou o F1 de alguns modelos, enquanto JPEG 30 reduziu o de outros; nao houve degradacao uniforme.

As tabelas espectrais mostram B1 a B4, E1 a E3, centroide e entropia por classe e cenario, bem como perfis radiais no treino. A comparacao de medias descreve distribuicoes e nao estabelece mecanismo causal. Os exemplos selecionados sao todas as imagens, em ordem do manifesto, com divergencia de acerto no original nas comparacoes espaciais versus hibridos aumentados.
