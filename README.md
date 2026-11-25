# Detecção de imagens geradas por IA

O dataset atual planejado é o CIFAKE, com resolução de 32 × 32 e seleção visual da categoria automobile. A seção CIFAKE ao final deste documento descreve o procedimento atual. As seções seguintes registram a preparação anterior do GenImage.

A etapa 1 prepara amostras do GenImage Stable Diffusion 1.4 para análise em CPU. A classe 0 corresponde a imagens do ImageNet e a classe 1 a imagens geradas pelo SD 1.4. A metodologia do projeto prevê análise por FFT 2D na faixa de 0,25 a 0,50 ciclos/pixel em etapas posteriores.

## Executar

Execute os comandos abaixo na raiz do projeto com Python 3.12 e PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m genimage_prep --profile smoke --source streaming
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
.\.venv\Scripts\python.exe -m genimage_prep --profile smoke --source local --local-root "data/Stable Diffusion V1.4"
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
- [results/dataset_summary.json](results/dataset_summary.json) registra a última execução, a procedência, as contagens, as distribuições e os detalhes de inválidos e duplicatas encontrados. O campo `status` distingue uma amostra concluída de uma tentativa interrompida.

`genimage_prep.images.preprocess_image` recebe bytes ou um caminho, decodifica a imagem, aplica a orientação EXIF, converte para RGB e retorna uma imagem Pillow de 256 × 256. O recorte central preserva a proporção e o redimensionamento usa Lanczos para ambas as classes. A função opera em memória e preserva os originais.

## Resultado da execução inicial

O `smoke` via streaming produziu 400 imagens a partir de 738.787.328 bytes de Arrow lidos. Foram observados zero arquivos inválidos e zero duplicatas exatas entre os candidatos inspecionados. As 200 imagens reais eram JPEG, com resoluções variadas. As 200 imagens geradas eram PNG de 512 × 512. Essa diferença pode introduzir um atalho de classificação por artefatos de compressão ou de redimensionamento, mesmo após a padronização.

O protocolo experimental do benchmark está descrito no [artigo do GenImage](https://arxiv.org/abs/2306.08571).

## Investigação de categoria única

O inventário preliminar em `results/category_inventory.csv` reúne as contagens por WNID no treino e por prefixo nos arquivos sintéticos das duas partições. A tabela do ImageNet associa nomes, WNIDs e índices. Ainda não foi identificada evidência de que o prefixo dos arquivos sintéticos use essa mesma ordem. Por isso as colunas de mapeamento confirmado e pares balanceados estão vazias. As imagens reais da avaliação têm identificadores sem WNID no índice disponível.

O `mapping.json` associa caminhos a índices globais de registros, mas não especifica offsets de bytes ou um índice de batches Arrow. A leitura de 64 KiB confirmou o esquema Arrow e revelou que o primeiro batch exige cerca de 7 MB. O acesso seletivo aos registros permanece por validar. A escolha da categoria e a amostra definitiva dependem da confirmação do mapeamento e da avaliação do custo de leitura.

Para reproduzir o inventário com os índices públicos já obtidos na pasta local `.venv`:

```powershell
.\.venv\Scripts\python.exe -m genimage_prep.category_inventory --class-map .venv/imagenet-class-index.json --train-index .venv/genimage-train-index.json --test-index .venv/genimage-test-index.json
```

## CIFAKE e seleção visual

O projeto passa a utilizar o CIFAKE para reduzir dificuldades de recuperação e de confirmação semântica encontradas na amostra GenImage. A versão 3 foi obtida no [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images). As imagens REAL procedem do CIFAR-10 e as FAKE foram geradas com Stable Diffusion 1.4. O projeto original fornece rótulos binários. [Repositório CIFAKE](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images) · [Artigo CIFAKE](https://doi.org/10.1109/ACCESS.2024.3356122)

Os nomes e diretórios inspecionados não fornecem confirmação semântica para automobile. As folhas em `results/cifake/contact_sheets/` contêm candidatos sorteados com seed 42, separados por classe e partição. Atribua `sim` na coluna `automobile` de `results/cifake/category_selection.csv` apenas quando a imagem mostrar um automóvel reconhecível. Use o mesmo critério para REAL e FAKE e deixe os casos ambíguos vazios. O par folha e posição identifica cada imagem. As folhas não constituem rótulos confirmados até a conferência. Sorteiam-se 1.500 candidatos por classe do treino e 500 por classe do teste. Se forem insuficientes, a preparação informará a falta.

Após a revisão, execute:

```powershell
.\.venv\Scripts\python.exe -m genimage_prep.cifake prepare
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

O manifesto resultante fica em `data/prepared/cifake/manifest.csv`. As cotas configuráveis em `config.yaml` são, por classe, 105 para treino e 15 para validação retiradas do treino oficial e 30 do teste oficial. Os arquivos selecionados mantêm seus bytes e são deduplicados por SHA-256. `preprocess_cifake` aplica orientação EXIF e conversão RGB, exige 32 × 32 e não redimensiona.

Para criar novamente as folhas em uma instalação limpa, extraia a versão 3 do arquivo oficial em `data/raw/cifake/` e execute `python -m genimage_prep.cifake inspect`. Esse comando protege o CSV de anotações existente contra sobrescrita. Um usuário autenticado pode obter o arquivo com a CLI Kaggle usando `kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -v 3`.

A metodologia futura utiliza FFT bidimensional, 16 intervalos radiais até 0,50 ciclos/pixel, cinco subfaixas entre 0,25 e 0,50, e Random Forest espacial, espectral e híbrido. Serão investigados JPEG 75 e 30, resize 32→24→32 e 32→16→32 e blur gaussiano com sigma 0,5 e 1,0. A resolução 32 × 32 limita a discriminação visual e a quantidade de frequências amostradas. O processo de geração e a seleção visual podem introduzir diferenças de conteúdo entre classes.

## Priorização semântica e revisão humana

A ferramenta auxiliar usa `cifar10_resnet20` pré-treinada em CIFAR-10, somente para priorizar a revisão de automóveis. A revisão do código é `786c16252c0fc58ee9adac063f8337cc4a7a497a`. Os pesos oficiais `cifar10_resnet20-4118986f.pt` têm SHA-256 `4118986f0df73003d572b0e397f0ac7b3f60af1f31aff3d2da164536e36f6ec8`. O código usado tem SHA-256 `885b8fd2ccfb8ee27c02d3a382e2951622fdcbc2af2bbca375aaf71d40f5f246`. A implementação verifica ambos os hashes. O treinamento publicado usa média RGB [0,4914, 0,4822, 0,4465] e desvio padrão [0,2023, 0,1994, 0,2010]. A imagem RGB permanece em 32 × 32, os valores são divididos por 255 e normalizados por canal. Automobile é a classe de índice 1. [Modelo e pesos oficiais](https://github.com/chenyaofo/pytorch-cifar-models/tree/786c16252c0fc58ee9adac063f8337cc4a7a497a) · [Configuração de treino](https://github.com/chenyaofo/image-classification-codebase/blob/master/conf/cifar10.conf)

Em uma instalação nova, instale as dependências opcionais e obtenha a revisão e os pesos. Os arquivos permanecem na pasta ignorada `.venv`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-semantic.txt
curl.exe -L https://codeload.github.com/chenyaofo/pytorch-cifar-models/zip/786c16252c0fc58ee9adac063f8337cc4a7a497a -o .venv/cifar-model-source.zip
Expand-Archive .venv/cifar-model-source.zip -DestinationPath .venv
curl.exe -L https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt -o .venv/cifar10_resnet20-4118986f.pt
.\.venv\Scripts\python.exe -m genimage_prep.semantic infer
.\.venv\Scripts\python.exe -m genimage_prep.semantic sheets
```

A inferência usa CPU, lotes, `eval()` e `inference_mode()`. Os resultados em `results/cifake/semantic_predictions.csv` são estimativas e incluem posição e probabilidade de automobile. As folhas priorizam as imagens que colocam automobile entre as duas primeiras classes, embaralhadas com seed 42. Uma folha adicional reúne 25 imagens aleatórias não priorizadas de cada combinação de classe e partição. As folhas misturam classes e não mostram origem ou confiança. O mapa de associação fica em `results/cifake/semantic_review_map.csv`. Para preservar a associação, a ferramenta recusa regenerar predições ou folhas enquanto o mapa existir.

Para revisar, execute `.\.venv\Scripts\python.exe -m genimage_prep.semantic review`. Abra no visualizador de imagens o caminho da folha exibida no terminal. Pressione Enter e informe os números dos automóveis de passeio reconhecíveis, separados por espaços ou vírgulas. Em seguida informe os números ambíguos. Os demais números recebem `nao`. Use `p` para pular e `q` para sair. Cada folha concluída é salva no CSV original e a execução seguinte retoma as folhas pendentes. Anotações existentes são preservadas. O comando `python -m genimage_prep.semantic status` informa as cotas ainda pendentes por classe e partição.

Aceite automóveis de passeio reconhecíveis. Caminhões, ônibus, motocicletas e veículos de outra categoria devem ser rejeitados. Marque como ambíguo quando não for possível identificar a categoria. Não use aparência realista, estética ou artefatos de IA como critério. Depois de confirmar pelo menos 120 imagens de cada classe no treino oficial e 30 de cada classe no teste oficial, execute `python -m genimage_prep.cifake prepare`.

A rede pode se comportar de modo diferente nas imagens reais e sintéticas. O treino do modelo em CIFAR-10 pode incluir imagens que originaram as imagens REAL de treinamento do CIFAKE. Suas previsões não constituem evidência independente da qualidade da seleção. A confiança da rede não deve ser usada como característica dos futuros detectores.
