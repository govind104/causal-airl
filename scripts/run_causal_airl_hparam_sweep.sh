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
LOG=results/logs/causal_airl_hparams.log
echo "========== [Causal-AIRL Experiments] ==========" > $LOG

# Run Causal-AIRL METHOD hyperparam sweep (kl_coeff, inv_coeff, latent_dim); Fixed train-Z
echo "========== [1] Running Causal-AIRL Hyperparam Sweep ==========" | tee -a $LOG

BASE_CFG="configs/causal_airl_ablation.yaml"
COMMON=( "-m" "experiments.sweeps"
         "--base" "${BASE_CFG}"
         "--save_root" "results/causal_airl_hparams"
         "--grid" "train.seed=42,123,456,789,2025"
         "--grid" "irl.kl_coeff=0.001,0.003,0.01"
         "--grid" "irl.inv_coeff=0.0,0.02,0.05"
         "--grid" "irl.latent_dim=2,4,8"
         "--grid" "irl.num_z_samples=1"
         "--grid" "irl.gamma=0.99"
         "--grid" "expert.num_trajectories=20"
         "--grid" "expert.confounder_value=0"
         "--grid" "env.name=ConfoundedGridWorld"
         "--grid" "env.slip_prob=0.0"
         "--grid" "env.reward_type=sparse" )

# Optional overrides from best hyperparams (shared training knobs)
OVRS=()
[ -n "$BEST_AIRL_ENTROPY" ] && OVRS+=( "--override" "irl.entropy_coef=${BEST_AIRL_ENTROPY}" )
[ -n "$BEST_AIRL_CLIP" ]    && OVRS+=( "--override" "irl.grad_clip_norm=${BEST_AIRL_CLIP}" )

$PY "${COMMON[@]}" "${OVRS[@]}" | tee -a $LOG

# Footer
echo "========== Causal-AIRL Experiments Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/causal_airl_hparams.log"
