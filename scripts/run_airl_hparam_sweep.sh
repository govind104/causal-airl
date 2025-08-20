#!/usr/bin/env bash

# --- Environment-agnostic Python resolver (Git Bash + Windows friendly) ---
: "${ENV_NAME:=causal-irl-env}"
resolve_py() {
  if command -v conda >/dev/null 2>&1; then
    echo "conda run -n ${ENV_NAME} python"; return
  fi
  if [ -n "${CONDA_PREFIX:-}" ]; then
    if command -v cygpath >/dev/null 2>&1; then
      local p="$(cygpath -u "$CONDA_PREFIX")/python.exe"
      [ -x "$p" ] && { echo "$p"; return; }
    fi
    local p="$CONDA_PREFIX/bin/python"
    [ -x "$p" ] && { echo "$p"; return; }
  fi
  echo "python"
}
PY="$(resolve_py)"

set -euo pipefail

# Setup log file
mkdir -p results/logs
LOG=results/logs/airl_hparams.log
echo "========== [AIRL Experiments] ==========" > $LOG

echo "========== [1] Running AIRL Hyperparam Sweep ==========" | tee -a $LOG

$PY -m experiments.sweeps \
  --base configs/airl_ablation.yaml \
  --save_root results/airl_hparams \
  --grid train.seed=42,123,456,789,2025 \
  --grid irl.entropy_coef=0.0,0.005,0.01 \
  --grid irl.grad_clip_norm=0.3,0.5,1.0 \
  --grid irl.gamma=0.99 \
  --grid expert.num_trajectories=20 \
  --grid env.slip_prob=0.0 \
  --grid env.reward_type=sparse | tee -a $LOG

echo "========== AIRL Experiments Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/airl_hparams.log"