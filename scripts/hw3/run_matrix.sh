#!/usr/bin/env bash
# HW3 core experiment matrix: E0-E3 x seeds 0, 1, 2 = 12 runs, HW2 protocol otherwise
# (60 epochs, batch 32, Adam lr 0.001, weight decay 1e-4, cosine, dropout 0.3, augmentation).
#
#   export MODELNET10_DATA_DIR=/path/to/modelnet10    # prepared cache (python -m app.classification.dataset prepare)
#   scripts/hw3/run_matrix.sh                          # all 12 runs, one after another
#   TASK=6 scripts/hw3/run_matrix.sh                   # run number 6 only (e.g. a Slurm array task)
#
# Run i trains condition i % 4 with seed i / 4: tasks 0-3 are E0-E3 with seed 0, and so on.
# Optional: PYTHON (default python), OUT (default backend/data/experiments/hw3),
# EPOCHS (default 60; smoke tests only), EXTRA_ARGS (e.g. --skip-test for a smoke test).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-python}
OUT=${OUT:-$ROOT/backend/data/experiments/hw3}
EPOCHS=${EPOCHS:-60}
CONDITIONS=(                                   # run id : model : hidden width
  "e0_max:mini_pointnet:64"
  "e1_fixed_dual:mini_pointnet_fixed_dual:64"
  "e2_gated_dual:mini_pointnet_gated_dual:64"
  "e3_max_h69:mini_pointnet:69"
)
SEEDS=(0 1 2)

run_task() {
  local task=$1 id model hidden seed
  IFS=: read -r id model hidden <<<"${CONDITIONS[$((task % 4))]}"
  seed=${SEEDS[$((task / 4))]}
  echo "task=$task condition=$id model=$model hidden=$hidden seed=$seed host=$(hostname) sha=$(git -C "$ROOT" rev-parse --short HEAD)"
  (cd "$ROOT/backend" && "$PYTHON" -m app.classification.training train \
      --model "$model" --hidden-dim "$hidden" --seed "$seed" --epochs "$EPOCHS" \
      --learning-rate 0.001 --batch-size 32 --weight-decay 0.0001 --scheduler cosine --dropout 0.3 \
      --activation relu --augment true --device cpu ${EXTRA_ARGS:-} \
      --out "$OUT/${id}_seed${seed}")
}

if [[ -n "${TASK:-}" ]]; then
  run_task "$TASK"
else
  for task in $(seq 0 11); do run_task "$task"; done
fi
