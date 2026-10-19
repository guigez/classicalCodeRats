# Classificação de intensidade de esforço em imagens termográficas de ratos

Pipeline de reprodução e extensão do trabalho de Bernini: duas famílias de
modelos (a CNN do artigo e uma ResNet50 com random forest), avaliadas com
particionamento **por sujeito**, nos quatro cenários de agrupamento de classes
(`multiclass`, `three_bernini`, `three_mlss`, `binary`).

## Estrutura

```
run_all.py            orquestrador: roda cada etapa como subprocesso
configs/              grids de hiperparametros de cada familia
src/
  manifest.py         1  inventario e validacao do metadata
  splits.py           2  particao por sujeito (sem vazamento, folds nao degenerados)
  dataset.py          3  acesso as imagens (PyTorch)
  models.py           4  BerniniCNN, ResNet50 + random forest
  metrics.py          5  metricas por quadro e por animal
  train.py            6  treino de um candidato
  grid_search.py      7  grid no fold, selecao na validacao, teste uma vez
  seed_replicates.py  7b retreino do vencedor com outras sementes
  explain_shap.py     8  atribuicao SHAP e enriquecimento animal/fundo
  aggregate.py        9  tabelas finais e comparacao pareada por animal
  common.py              utilitarios compartilhados
tests/                testes (pytest)
```

## Instalação

```bash
pip install -r requirements.txt
```

Os dados não estão no repositório: `metadata.csv` (colunas `filename`,
`mice_id`, `class`) e a pasta de imagens precisam ser colocados na raiz.

## Execução

O desenho usado na dissertação (mesmos animais de teste em todos os cenários):

```bash
python run_all.py --task binary --split_mode fixed_paired \
    --fixed_test_subjects 1 9 10 11 --images_dir imagens --seeds 1 2 3 4 5
```

No servidor, com a estrutura `Bernini/_dataset/{images,metadata-1.csv}`,
`Bernini/Classical/classicalCodeRats/` (este repositório) e saída em
`Bernini/Classical/main_results/`, os quatro cenários rodam em sequência sem
sessão aberta:

```bash
nohup bash scripts/run_server.sh > /dev/null 2>&1 &
tail -f ../main_results/run_server.log
```

As opções (cenários, sementes, dispositivo, caminhos) estão no cabeçalho de
`scripts/run_server.sh`.

Cada etapa grava um `params.json`; rodar de novo só refaz o que mudou
(`--force` refaz tudo). Resultados em `<results_dir>/<task>/` (padrão `resultados/`), tabelas finais em
`<results_dir>/<task>/_tabelas/`:

| arquivo | conteúdo |
|---|---|
| `paired_by_subject.csv` | métrica de cada braço por animal de teste e diferenças pareadas |
| `paired_comparison.csv` | direção (vitórias), tamanho do efeito, teste de sinal e piso do p |
| `seed_summary_by_subject.csv` | média e desvio entre sementes, por animal e braço |
| `seed_paired_by_subject.csv` | diferença entre braços comparada ao ruído entre sementes |
| `aggregate_metrics.csv` | média, desvio e extremos de cada métrica entre folds |

## Testes

```bash
python -m pytest tests
```
