# Detecção de imagens geradas por IA

O dataset atual planejado é o CIFAKE, com resolução de 32 × 32 e seleção visual da categoria automobile. A seção CIFAKE ao final deste documento descreve o procedimento atual. As seções seguintes registram a preparação anterior do GenImage.

A etapa 1 prepara amostras do GenImage Stable Diffusion 1.4 para análise em CPU. A classe 0 corresponde a imagens do ImageNet e a classe 1 a imagens geradas pelo SD 1.4. A metodologia do projeto prevê análise por FFT 2D na faixa de 0,25 a 0,50 ciclos/pixel em etapas posteriores.

## Executar

Execute os comandos abaixo na raiz do projeto com Python 3.12 e PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s testes -v
.\.venv\Scripts\python.exe -m preparacao_dados --profile smoke --source streaming
```

Para preparar a amostra de 4.000 imagens, troque `--profile smoke` por `--profile experiment`. A seed, as cotas por classe e os limites de leitura ficam em [config.yaml](config.yaml).

| Perfil | Treino por classe | Validação por classe | Teste por classe | Total |
| --- | ---: | ---: | ---: | ---: |
| `smoke` | 140 | 20 | 40 | 400 |
| `experiment` | 1.400 | 200 | 400 | 4.000 |

## Fontes de dados

O streaming usa a revisão `3f4b9f921a673be09a93b335ed728cea0c6ecf33` do espelho Hugging Face. A configuração `sd14-train` tem 189 shards (93,06 GB) e `sd14-test` tem 8 (3,56 GB). Os campos são `image_path`, bytes da imagem, classe e gerador. O treino traz também MD5, largura e altura de origem. O programa mede formato e dimensões a partir dos bytes decodificados. A configuração `sd14-validation` do espelho é um alias do teste. [Espelho e esquema no Hugging Face](https://huggingface.co/datasets/nebula/GenImage-arrow/tree/3f4b9f921a673be09a93b335ed728cea0c6ecf33)

A leitura usa requisições HTTP parciais com limite total de 1.024 MiB para `smoke` e 5.120 MiB para `experiment`. O programa verifica o espaço livre antes de conectar, descarta os lotes Arrow após o uso e grava somente as imagens selecionadas. Se o servidor recusar a resposta parcial ou a amostra ficar aquém das cotas, o processo termina com erro e registra o resultado da tentativa.

Para usar a distribuição oficial, extraia apenas o subconjunto SD 1.4 em uma pasta com `train/ai`, `train/nature`, `val/ai` e `val/nature`. Verifique o espaço necessário para o arquivo compactado e a extração antes de baixá-lo. [Repositório oficial e links de download do GenImage](https://github.com/GenImage-Dataset/GenImage)

```powershell
.\.venv\Scripts\python.exe -m preparacao_dados --profile smoke --source local --local-root "data/Stable Diffusion V1.4"
```

Use `--local-version` para registrar a versão ou o SHA-256 do arquivo oficial extraído. O modo local embaralha o inventário inteiro de caminhos elegíveis de cada classe antes de ler os arquivos.

## Amostragem e integridade

- A validação experimental é retirada do treino original. O teste experimental usa exclusivamente a avaliação original, chamada `val` nos arquivos oficiais e `test` no espelho. O manifesto usa `test` como nome normalizado.
- Com seed 42, o streaming embaralha os shards e as linhas dos lotes lidos. As cotas por shard incentivam variedade de origens. A leitura alcança até três lotes por shard no `smoke` e 12 no `experiment`. Como os lotes são lidos do começo de cada shard, a amostra tem probabilidades de inclusão desiguais. O relatório identifica os shards que contribuíram.
- O teste é preenchido primeiro. Um SHA-256 já aceito em qualquer conjunto faz a cópia posterior ser descartada, inclusive quando os rótulos divergem. Os números de inválidos e duplicatas se referem aos candidatos inspecionados. Alterações de codificação ou de conteúdo podem produzir hashes diferentes para imagens visualmente semelhantes.
- O relatório conta categorias por classe e split quando há um WNID do ImageNet no caminho ou um índice semântico no nome da imagem gerada. WNIDs e índices do SD 1.4 ficam em espaços de identificação distintos. Os nomes das imagens reais de avaliação contêm apenas o número da imagem e ficam com categoria vazia. Caminhos, formatos e demais metadados servem à auditoria. As etapas futuras de classificação devem consumir os pixels padronizados.

## Saídas

- `data/prepared/<perfil>/manifest.csv` lista ID na fonte, caminho relativo ao manifesto, classe, origem, splits, categoria quando disponível, largura, altura, formato e SHA-256.
- `data/prepared/<perfil>/originals/` preserva os bytes fornecidos pela fonte, sem recompressão.
- [resultados/resumo_dataset.json](resultados/resumo_dataset.json) registra a última execução, a procedência, as contagens, as distribuições e os detalhes de inválidos e duplicatas encontrados. O campo `status` distingue uma amostra concluída de uma tentativa interrompida.

`preparacao_dados.imagens.preprocess_image` recebe bytes ou um caminho, decodifica a imagem, aplica a orientação EXIF, converte para RGB e retorna uma imagem Pillow de 256 × 256. O recorte central preserva a proporção e o redimensionamento usa Lanczos para ambas as classes. A função opera em memória e preserva os originais.

## Resultado da execução inicial

O `smoke` via streaming produziu 400 imagens a partir de 738.787.328 bytes de Arrow lidos. Foram observados zero arquivos inválidos e zero duplicatas exatas entre os candidatos inspecionados. As 200 imagens reais eram JPEG, com resoluções variadas. As 200 imagens geradas eram PNG de 512 × 512. Essa diferença pode introduzir um atalho de classificação por artefatos de compressão ou de redimensionamento, mesmo após a padronização.

O protocolo experimental do benchmark está descrito no [artigo do GenImage](https://arxiv.org/abs/2306.08571).

## Investigação de categoria única

O inventário preliminar em `resultados/inventario_categorias.csv` reúne as contagens por WNID no treino e por prefixo nos arquivos sintéticos das duas partições. A tabela do ImageNet associa nomes, WNIDs e índices. Ainda não foi identificada evidência de que o prefixo dos arquivos sintéticos use essa mesma ordem. Por isso as colunas de mapeamento confirmado e pares balanceados estão vazias. As imagens reais da avaliação têm identificadores sem WNID no índice disponível.

O `mapping.json` associa caminhos a índices globais de registros, mas não especifica offsets de bytes ou um índice de batches Arrow. A leitura de 64 KiB confirmou o esquema Arrow e revelou que o primeiro batch exige cerca de 7 MB. O acesso seletivo aos registros permanece por validar. A escolha da categoria e a amostra definitiva dependem da confirmação do mapeamento e da avaliação do custo de leitura.

Para reproduzir o inventário com os índices públicos já obtidos na pasta local `.venv`:

```powershell
.\.venv\Scripts\python.exe -m preparacao_dados.inventario_categorias --class-map .venv/imagenet-class-index.json --train-index .venv/genimage-train-index.json --test-index .venv/genimage-test-index.json
```

## CIFAKE e seleção visual

O projeto passa a utilizar o CIFAKE para reduzir dificuldades de recuperação e de confirmação semântica encontradas na amostra GenImage. A versão 3 foi obtida no [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images). As imagens REAL procedem do CIFAR-10 e as FAKE foram geradas com Stable Diffusion 1.4. O projeto original fornece rótulos binários. [Repositório CIFAKE](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images) · [Artigo CIFAKE](https://doi.org/10.1109/ACCESS.2024.3356122)

Os nomes e diretórios inspecionados não fornecem confirmação semântica para automobile. As folhas em `resultados/cifake/folhas_contato/` contêm candidatos sorteados com seed 42, separados por classe e partição. Atribua `sim` na coluna `automobile` de `resultados/cifake/selecao_categoria.csv` apenas quando a imagem mostrar um automóvel reconhecível. Use o mesmo critério para REAL e FAKE e deixe os casos ambíguos vazios. O par folha e posição identifica cada imagem. As folhas não constituem rótulos confirmados até a conferência. Sorteiam-se 1.500 candidatos por classe do treino e 500 por classe do teste. Se forem insuficientes, a preparação informará a falta.

Após a revisão, execute:

```powershell
.\.venv\Scripts\python.exe -m preparacao_dados.cifake prepare
.\.venv\Scripts\python.exe -m unittest discover -s testes -v
```

O manifesto resultante fica em `data/preparados/cifake/manifest.csv`. As cotas configuráveis em `config.yaml` são, por classe, 105 para treino e 15 para validação retiradas do treino oficial e 30 do teste oficial. Os arquivos selecionados mantêm seus bytes e são deduplicados por SHA-256. `preprocess_cifake` aplica orientação EXIF e conversão RGB, exige 32 × 32 e não redimensiona.

Para criar novamente as folhas em uma instalação limpa, extraia a versão 3 do arquivo oficial em `data/raw/cifake/` e execute `python -m preparacao_dados.cifake inspect`. Esse comando protege o CSV de anotações existente contra sobrescrita. Um usuário autenticado pode obter o arquivo com a CLI Kaggle usando `kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -v 3`.

A metodologia futura utiliza FFT bidimensional, 16 intervalos radiais até 0,50 ciclos/pixel, cinco subfaixas entre 0,25 e 0,50, e Random Forest espacial, espectral e híbrido. Serão investigados JPEG 75 e 30, resize 32→24→32 e 32→16→32 e blur gaussiano com sigma 0,5 e 1,0. A resolução 32 × 32 limita a discriminação visual e a quantidade de frequências amostradas. O processo de geração e a seleção visual podem introduzir diferenças de conteúdo entre classes.

## Priorização semântica e revisão humana

A ferramenta auxiliar usa `cifar10_resnet20` pré-treinada em CIFAR-10, somente para priorizar a revisão de automóveis. A revisão do código é `786c16252c0fc58ee9adac063f8337cc4a7a497a`. Os pesos oficiais `cifar10_resnet20-4118986f.pt` têm SHA-256 `4118986f0df73003d572b0e397f0ac7b3f60af1f31aff3d2da164536e36f6ec8`. O código usado tem SHA-256 `885b8fd2ccfb8ee27c02d3a382e2951622fdcbc2af2bbca375aaf71d40f5f246`. A implementação verifica ambos os hashes. O treinamento publicado usa média RGB [0,4914, 0,4822, 0,4465] e desvio padrão [0,2023, 0,1994, 0,2010]. A imagem RGB permanece em 32 × 32, os valores são divididos por 255 e normalizados por canal. Automobile é a classe de índice 1. [Modelo e pesos oficiais](https://github.com/chenyaofo/pytorch-cifar-models/tree/786c16252c0fc58ee9adac063f8337cc4a7a497a) · [Configuração de treino](https://github.com/chenyaofo/image-classification-codebase/blob/master/conf/cifar10.conf)

Em uma instalação nova, instale as dependências opcionais e obtenha a revisão e os pesos. Os arquivos permanecem na pasta ignorada `.venv`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-semantic.txt
curl.exe -L https://codeload.github.com/chenyaofo/pytorch-cifar-models/zip/786c16252c0fc58ee9adac063f8337cc4a7a497a -o .venv/cifar-model-source.zip
Expand-Archive .venv/cifar-model-source.zip -DestinationPath .venv
curl.exe -L https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt -o .venv/cifar10_resnet20-4118986f.pt
.\.venv\Scripts\python.exe -m preparacao_dados.selecao_semantica infer
.\.venv\Scripts\python.exe -m preparacao_dados.selecao_semantica sheets
```

A inferência usa CPU, lotes, `eval()` e `inference_mode()`. Os resultados em `resultados/cifake/predicoes_semanticas.csv` são estimativas e incluem posição e probabilidade de automobile. As folhas priorizam as imagens que colocam automobile entre as duas primeiras classes, embaralhadas com seed 42. Uma folha adicional reúne 25 imagens aleatórias não priorizadas de cada combinação de classe e partição. As folhas misturam classes e não mostram origem ou confiança. O mapa de associação fica em `resultados/cifake/mapa_revisao_semantica.csv`. Para preservar a associação, a ferramenta recusa regenerar predições ou folhas enquanto o mapa existir.

Para revisar, execute `.\.venv\Scripts\python.exe -m preparacao_dados.selecao_semantica review`. Abra no visualizador de imagens o caminho da folha exibida no terminal. Pressione Enter e informe os números dos automóveis de passeio reconhecíveis, separados por espaços ou vírgulas. Em seguida informe os números ambíguos. Os demais números recebem `nao`. Use `p` para pular e `q` para sair. Cada folha concluída é salva no CSV original e a execução seguinte retoma as folhas pendentes. Anotações existentes são preservadas. O comando `python -m preparacao_dados.selecao_semantica status` informa as cotas ainda pendentes por classe e partição.

Aceite automóveis de passeio reconhecíveis. Caminhões, ônibus, motocicletas e veículos de outra categoria devem ser rejeitados. Marque como ambíguo quando não for possível identificar a categoria. Não use aparência realista, estética ou artefatos de IA como critério. Depois de confirmar pelo menos 120 imagens de cada classe no treino oficial e 30 de cada classe no teste oficial, execute `python -m preparacao_dados.cifake prepare`.

A rede pode se comportar de modo diferente nas imagens reais e sintéticas. O treino do modelo em CIFAR-10 pode incluir imagens que originaram as imagens REAL de treinamento do CIFAKE. Suas previsões não constituem evidência independente da qualidade da seleção. A confiança da rede não deve ser usada como característica dos futuros detectores.

## Análise exploratória de frequência

A análise usa exclusivamente as 210 imagens do treino do manifesto CIFAKE. Para reproduzir:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s testes -v
.\.venv\Scripts\python.exe -m analise_frequencias.explorar
```

Cada imagem é decodificada em RGB a 32 × 32, sem interpolação. A escala de cinza usa Y = 0,299 R + 0,587 G + 0,114 B, com canais float64 divididos por 255. Subtrai-se a média de Y e calcula-se F = fftshift(fft2(Y − média)). A potência é P = |F|². A grade utiliza fftfreq em ciclos/pixel, deslocada com fftshift, e r = √(fx² + fy²). A região analisada contém 0 < r ≤ 0,50. Excluem-se DC e os cantos com r > 0,50. A potência normalizada é P dividida pela soma de P na região, em escala linear.

O perfil tem 16 intervalos de largura 1/32 entre 0 e 0,50. Cada intervalo inclui o limite inferior e exclui o superior, exceto o último, que inclui 0,50. O primeiro intervalo fica vazio porque DC é removida. Os campos desse intervalo ficam vazios no CSV e nulos nas estatísticas, sem dividir por zero. Imagens de energia nula são registradas no relatório e excluídas das estatísticas. A potência radial média é a soma da potência normalizada dos coeficientes de um intervalo dividida pela quantidade desses coeficientes. A energia integrada é essa soma, sem a divisão pela quantidade.

As energias somam coeficientes normalizados nas faixas E1 [0,25; 0,3125), E2 [0,3125; 0,375) e E3 [0,375; 0,50]. Etotal é a soma para [0,25; 0,50], e Ebelow corresponde a 0 < r < 0,25. O programa verifica a igualdade entre Etotal e E1 + E2 + E3. As estatísticas de classe usam desvio padrão amostral e quartis por interpolação linear.

O controle pareado aplica JPEG em memória após a decodificação RGB, com qualidade 90, subsampling 0 (4:4:4), optimize e progressive desativados. Os arquivos originais são preservados. A recompressão não elimina necessariamente os efeitos da compressão histórica e não demonstra causalidade.

`resultados/frequencias/` contém características por imagem, perfis individuais, estatísticas por classe e cenário e `resumo.json`, com hash do manifesto e parâmetros. Os sete gráficos em `figuras/` mostram exemplos e FFT, espectros bidimensionais médios, perfis radiais com dispersão, energias médias, boxplots, dispersão E1 × E3 e o controle JPEG. Espectros dos exemplos usam log(1 + potência) somente na visualização. O perfil e o espectro médio bidimensional exibem potência normalizada linear.

A resolução limita a grade a passos de 1/32 ciclos/pixel e produz poucos coeficientes nas faixas. Sem janela espectral, as bordas podem causar leakage. A construção do CIFAKE, a compressão JPEG anterior e a seleção visual auxiliada por classificador podem introduzir diferenças entre classes. A análise é descritiva e usa apenas o treino para evitar seleção de parâmetros pelo teste.


## Etapa 3: extração de características

Estrutura principal: `preparacao_dados/` (preparação e seleção), `analise_frequencias/` (FFT e exploração), `extracao_caracteristicas/` (espaciais, espectrais e execução), `testes/` e `resultados/caracteristicas/`. Os dados brutos e o manifesto histórico GenImage mantêm seus caminhos originais.

| Antes | Agora |
| --- | --- |
| `genimage_prep/` | `preparacao_dados/` |
| `images.py`, `prepare.py`, `sources.py` | `imagens.py`, `preparar.py`, `fontes.py` |
| `category_inventory.py`, `semantic.py` | `inventario_categorias.py`, `selecao_semantica.py` |
| `frequency_analysis/`, `spectrum.py`, `explore.py` | `analise_frequencias/`, `espectro.py`, `explorar.py` |
| `test_category_inventory.py`, `test_prepare.py`, `test_semantic.py`, `test_spectrum.py` | `test_inventario_categorias.py`, `test_preparacao.py`, `test_selecao_semantica.py`, `test_espectro.py` |

Execute na raiz, depois de instalar `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s testes -v
.\.venv\Scripts\python.exe -m extracao_caracteristicas.extrair
.\.venv\Scripts\python.exe -m extracao_caracteristicas.extrair --saida resultados/caracteristicas_verificacao
```

O módulo `extracao_caracteristicas.extrair(imagem)` recebe uma imagem Pillow ou array RGB 32 × 32: inteiros entre 0 e 255 ou float entre 0 e 1. Não redimensiona nem altera os arquivos. A ordem fixa está nas constantes `NOMES_ESPACIAIS`, `NOMES_ESPECTRAIS` e `NOMES_HIBRIDOS` e no relatório.

As 40 características espaciais são: médias RGB (3), desvios padrão populacionais RGB (3), histogramas RGB (24; oito intervalos de largura 1/8, último incluindo 1, divididos por 1024), correlações Pearson RG, RB, GB (3), médias de diferenças absolutas horizontais e verticais por canal (6), e variância populacional do Laplaciano (1). Se um canal é constante, a correlação envolvendo-o é definida como zero. Diferenças usam somente vizinhos existentes. O Laplaciano usa cinza Y = 0,299 R + 0,587 G + 0,114 B na escala [0,1], kernel `[[0,1,0],[1,-4,1],[0,1,0]]` e extensão por repetição da borda.

As 3 características espectrais E1, E2 e E3 reutilizam a FFT e as faixas da Etapa 2 descritas acima. Em energia nula, a FFT anterior retorna `None`; o vetor espectral aqui é definido como zero e os IDs são registrados no relatório. Etotal não é entrada porque é exatamente E1 + E2 + E3; Ebelow também é dependente quando a energia normalizada é não nula. A abordagem híbrida concatena as 40 espaciais e as 3 espectrais, nessa ordem.

Os três CSVs em `resultados/caracteristicas/` preservam a ordem do manifesto e contêm `image_id`, `class`, `experimental_split` e apenas as características da abordagem. Esses três metadados não são entradas para o classificador. O relatório JSON registra parâmetros, dimensões, hash, verificações, energias nulas e pares com |Pearson| ≥ 0,95; `estatisticas_treino.csv` resume somente o treino. Não há eliminação automática, normalização ajustada, comparação de classes na validação nem treinamento. As 60 imagens de teste continuam reservadas. As características são sensíveis ao conteúdo e à compressão, o conjunto é pequeno e limitado a automóveis CIFAKE, e correlações observadas no treino não demonstram generalização. Qualquer ajuste futuro deve usar exclusivamente treino e evitar vazamento entre partições.

O relatório de preparação registra agora 359 anotações `sim` no CSV de revisão, das quais 300 foram selecionadas; o valor anteriormente registrado (zero) é preservado como histórico.


## Caminhos após a padronização

```text
preparacao_dados/       imagens.py, preparar.py, fontes.py, inventario_categorias.py, selecao_semantica.py, cifake.py
analise_frequencias/    espectro.py, explorar.py
extracao_caracteristicas/  espaciais.py, espectrais.py, extrair.py
testes/                 test_*.py
data/raw/cifake/        dados originais referenciados pelos CSVs históricos
data/prepared/smoke/    manifesto e imagens históricos GenImage
data/preparados/cifake/ manifest.csv e originals/
resultados/cifake/      seleção, inventário, predições, mapa e folhas
resultados/frequencias/ CSVs, resumo.json e figuras/
resultados/caracteristicas/ CSVs e relatorio.json
```

Os caminhos `data/raw/` constam nos CSVs de seleção, predição e revisão e foram mantidos para que continuem resolvíveis. O resumo histórico GenImage aponta para `data/prepared/smoke/`; esse caminho também permanece. `originals/` é parte dos caminhos relativos dos manifestos, cujos bytes não foram modificados. Por isso a raiz `data/` permanece ignorada pelo Git. O manifesto CIFAKE foi movido junto de sua pasta sem alteração de conteúdo. Nomes em inglês registrados nos CSVs, rótulos oficiais e resultados históricos não foram traduzidos.
