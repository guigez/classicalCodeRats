#!/usr/bin/env bash
# Roda os quatro cenarios no servidor, em sequencia, sem depender da sessao
# SSH: lance com nohup e pode desconectar.
#
# Estrutura esperada (caminhos derivados da posicao deste script):
#   Bernini/_dataset/images/            imagens
#   Bernini/_dataset/metadata-1.csv     metadata (filename, mice_id, class)
#   Bernini/Classical/classicalCodeRats/   este repositorio
#   Bernini/Classical/main_results/        saida (criada se nao existir)
#
# Uso, com o ambiente Python ja ativado:
#   nohup bash scripts/run_server.sh > /dev/null 2>&1 &
#
# Acompanhar:   tail -f ../main_results/run_server.log
# Interromper:  kill -- -$(cat ../main_results/run_server.pid)
#
# Variaveis opcionais (ex.: TASKS="binary" SEEDS="1 2 3" nohup bash ...):
#   PYTHON      interpretador                  (padrao: python do PATH)
#   TASKS       cenarios, na ordem             (padrao: os quatro)
#   SEEDS       sementes das replicas          (padrao: 1 2 3 4 5; "" desliga)
#   FIXED_TEST  animais de teste               (padrao: 1 9 10 11)
#   DEVICE      auto | cuda | cpu              (padrao: auto)
#   DATASET     pasta dos dados                (padrao: Bernini/_dataset)
#   RESULTS     pasta de saida                 (padrao: Bernini/Classical/main_results)
#   EXTRA_ARGS  argumentos a mais para run_all.py (ex.: "--force")

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLASSICAL="$(dirname "$REPO")"
BERNINI="$(dirname "$CLASSICAL")"

DATASET="${DATASET:-$BERNINI/_dataset}"
RESULTS="${RESULTS:-$CLASSICAL/main_results}"
PYTHON="${PYTHON:-python}"
TASKS="${TASKS:-binary three_bernini three_mlss multiclass}"
SEEDS="${SEEDS-1 2 3 4 5}"
FIXED_TEST="${FIXED_TEST:-1 9 10 11}"
DEVICE="${DEVICE:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$RESULTS"
LOG="$RESULTS/run_server.log"
exec >>"$LOG" 2>&1
echo $$ > "$RESULTS/run_server.pid"

echo "======================================================================"
echo "inicio  $(date '+%F %T')   commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "dados   $DATASET"
echo "saida   $RESULTS"
echo "python  $(command -v "$PYTHON")"
echo "tarefas $TASKS   sementes: ${SEEDS:-nenhuma}   teste: $FIXED_TEST   device: $DEVICE"
echo "======================================================================"

if [ ! -d "$DATASET/images" ]; then echo "ERRO: $DATASET/images nao existe"; exit 1; fi
if [ ! -f "$DATASET/metadata-1.csv" ]; then echo "ERRO: $DATASET/metadata-1.csv nao existe"; exit 1; fi
if ! "$PYTHON" -c "import torch, torchvision, sklearn, shap" ; then
    echo "ERRO: dependencias ausentes em $(command -v "$PYTHON"); ative o ambiente e rode pip install -r requirements.txt"
    exit 1
fi

cd "$REPO" || exit 1
seeds_arg=()
if [ -n "$SEEDS" ]; then seeds_arg=(--seeds $SEEDS); fi

falhas=()
for task in $TASKS; do
    echo
    echo "------ $task  ($(date '+%F %T')) ------"
    # shellcheck disable=SC2086
    if ! "$PYTHON" run_all.py --task "$task" \
            --split_mode fixed_paired --fixed_test_subjects $FIXED_TEST \
            --images_dir "$DATASET/images" --metadata "$DATASET/metadata-1.csv" \
            --results_dir "$RESULTS" --device "$DEVICE" \
            "${seeds_arg[@]}" $EXTRA_ARGS; then
        falhas+=("$task")
    fi
done

echo
echo "======================================================================"
echo "fim     $(date '+%F %T')"
if [ ${#falhas[@]} -gt 0 ]; then
    echo "tarefas com falha: ${falhas[*]}  (rode de novo: as etapas concluidas sao puladas)"
else
    echo "todas as tarefas concluidas sem falhas"
fi
echo "tabelas: $RESULTS/<tarefa>/_tabelas/"
echo "======================================================================"
rm -f "$RESULTS/run_server.pid"
[ ${#falhas[@]} -eq 0 ]
