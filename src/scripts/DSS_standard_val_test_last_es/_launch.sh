#!/usr/bin/env bash
set -euo pipefail
if (($# < 2)); then
  echo "Usage: $0 CANCER MODEL [run_survival.py options]" >&2
  exit 2
fi
cancer=$1
model=$2
shift 2
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
export RESULTS_ROOT="${RESULTS_ROOT:-${repo_root}/results/DSS_standard_val_test_last_es}"
python_bin=${PYTHON_BIN:-/data2/lama/miniconda3/envs/MIL/bin/python}
args=(
  --cancer "${cancer}" --model "${model}"
  --endpoint "${SURVIVAL_ENDPOINT:-dss}" --split-mode "${SPLIT_MODE:-train_test}"
  --loss "${LOSS_FN:-nll}" --checkpoint "${CHECKPOINT_SELECTION:-last}"
  --checkpoint-metric "${CHECKPOINT_METRIC:-c_index}"
  --early-stopping "${EARLY_STOPPING:-1}"
  --es-min-epochs "${ES_MIN_EPOCHS:-3}" --es-patience "${ES_PATIENCE:-5}"
  --es-metric "${ES_METRIC:-loss}" --max-epochs "${MAX_EPOCHS:-10}"
  --lr "${LR:-1e-4}" --seed "${SEED:-1}" --folds "${FOLDS:-0,1,2,3,4}"
  --num-workers "${NUM_WORKERS:-2}" --train-bag-size "${TRAIN_BAG_SIZE:-4096}"
  --starpath-patches "${STARPATH_PATCHES_PER_SLIDE:-512}"
  --inject-layers "${STARPATH_TITAN_INJECT_LAYERS:-2,4}"
  --trainable-layers "${STARPATH_TITAN_TRAINABLE_LAYERS:-2,3,4,5}"
)
[[ -z ${BATCH_SIZE:-} ]] || args+=(--batch-size "${BATCH_SIZE}")
[[ -z ${RNA_SET:-} ]] || args+=(--rna-set "${RNA_SET}")
[[ -z ${DATA_CONFIG:-} ]] || args+=(--data-config "${DATA_CONFIG}")
[[ -z ${RESULTS_ROOT:-} ]] || args+=(--results-root "${RESULTS_ROOT}")
[[ -z ${RUN_ID:-} ]] || args+=(--run-id "${RUN_ID}")
[[ ${DRY_RUN:-0} != 1 ]] || args+=(--dry-run)
exec "${python_bin}" "${repo_root}/tools/run_survival.py" "${args[@]}" "$@"
