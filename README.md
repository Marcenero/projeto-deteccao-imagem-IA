# Detecção de imagens geradas por IA

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
