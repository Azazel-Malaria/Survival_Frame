#!/usr/bin/env bash
set -euo pipefail
if (($# < 1)); then
  echo "Usage: $0 BRCA|BLCA|STAD|HNSC|LUAD|LUSC|CRC|KIRC [prototype options]" >&2
  exit 2
fi
cancer=$1
shift
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
python_bin=${PYTHON_BIN:-/data2/lama/miniconda3/envs/MIL/bin/python}
args=(--cancer "${cancer}" --endpoint "${SURVIVAL_ENDPOINT:-dss}"
      --split-mode "${SPLIT_MODE:-train_test}" --folds "${FOLDS:-0,1,2,3,4}")
[[ -z ${DATA_CONFIG:-} ]] || args+=(--data-config "${DATA_CONFIG}")
[[ ${DRY_RUN:-0} != 1 ]] || args+=(--dry-run)
exec "${python_bin}" "${repo_root}/tools/build_prototypes.py" "${args[@]}" "$@"
