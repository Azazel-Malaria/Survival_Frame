#!/usr/bin/env bash
set -euo pipefail
# Set these variables here or in the environment. Explicit CLI options override them.
export SURVIVAL_ENDPOINT="${SURVIVAL_ENDPOINT:-os}"
export SPLIT_MODE="${SPLIT_MODE:-train_test}"
export LOSS_FN="${LOSS_FN:-nll}"
export CHECKPOINT_SELECTION="${CHECKPOINT_SELECTION:-last}"
export EARLY_STOPPING="${EARLY_STOPPING:-0}"
export MAX_EPOCHS="${MAX_EPOCHS:-10}"
export LR="${LR:-1e-4}"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${script_dir}/../_launch.sh" LUAD slotspe "$@"
