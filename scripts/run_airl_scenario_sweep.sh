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

# Optional best hyperparams (export before running; empty = no override)
: "${BEST_AIRL_ENTROPY:=0.005}"
: "${BEST_AIRL_CLIP:=1.0}"

# Setup log file
mkdir -p results/logs
LOG=results/logs/airl_scenarios.log
echo "========== [AIRL Experiments] ==========" > $LOG

# Run SCENARIO sweep (γ, N, slip, reward_type)
echo "========== [1] Running AIRL Scenario Sweep ==========" | tee -a $LOG

BASE_CFG="configs/airl_ablation.yaml"
COMMON=( "-m" "experiments.sweeps"
         "--base" "${BASE_CFG}"
         "--save_root" "results/airl_scenarios"
         "--grid" "train.seed=42,123,456,789,2025"
         "--grid" "irl.gamma=0.9,0.95,0.99"
         "--grid" "expert.num_trajectories=5,10,20,50"
         "--grid" "env.slip_prob=0.0,0.1,0.2"
         "--grid" "env.reward_type=sparse,shaped" )

if [ -n "$BEST_AIRL_ENTROPY" ]; then COMMON+=( "--grid" "irl.entropy_coef=${BEST_AIRL_ENTROPY}" ); fi
if [ -n "$BEST_AIRL_CLIP" ];    then COMMON+=( "--grid" "irl.grad_clip_norm=${BEST_AIRL_CLIP}" ); fi

$PY "${COMMON[@]}" | tee -a $LOG

# Footer
echo "========== AIRL Experiments Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/airl_scenarios.log"
