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
: "${BEST_CAIRL_KL:=0.001}"
: "${BEST_CAIRL_INV:=0.02}"
: "${BEST_CAIRL_LATENT:=8}"

# Setup log file
mkdir -p results/logs
LOG=results/logs/causal_airl_scenarios.log
echo "========== [Causal-AIRL Experiments] ==========" > $LOG

# Run SCENARIO sweep (γ, N, slip, reward_type)
echo "========== [1] Running Causal-AIRL Scenario Sweep ==========" | tee -a $LOG

BASE_CFG="configs/causal_airl_ablation.yaml"
COMMON=( "-m" "experiments.sweeps"
         "--base" "${BASE_CFG}"
         "--save_root" "results/causal_airl_scenarios"
         "--grid" "train.seed=42,123,456,789,2025"
         "--grid" "irl.gamma=0.9,0.95,0.99"
         "--grid" "expert.num_trajectories=5,10,20,50"
         "--grid" "env.slip_prob=0.0,0.1,0.2"
         "--grid" "env.reward_type=sparse,shaped"
         "--grid" "irl.num_z_samples=1,5" )

OVRS=()
[ -n "$BEST_AIRL_ENTROPY" ] && OVRS+=( "--override" "irl.entropy_coef=${BEST_AIRL_ENTROPY}" )
[ -n "$BEST_AIRL_CLIP" ]    && OVRS+=( "--override" "irl.grad_clip_norm=${BEST_AIRL_CLIP}" )
[ -n "$BEST_CAIRL_KL" ]     && OVRS+=( "--override" "irl.kl_coeff=${BEST_CAIRL_KL}" )
[ -n "$BEST_CAIRL_INV" ]    && OVRS+=( "--override" "irl.inv_coeff=${BEST_CAIRL_INV}" )
[ -n "$BEST_CAIRL_LATENT" ] && OVRS+=( "--override" "irl.latent_dim=${BEST_CAIRL_LATENT}" )

if [ -n "$BEST_AIRL_ENTROPY" ]; then COMMON+=( "--grid" "irl.entropy_coef=${BEST_AIRL_ENTROPY}" ); fi
if [ -n "$BEST_AIRL_CLIP" ];    then COMMON+=( "--grid" "irl.grad_clip_norm=${BEST_AIRL_CLIP}" ); fi
if [ -n "$BEST_CAIRL_KL" ]; then COMMON+=( "--grid" "irl.kl_coeff=${BEST_CAIRL_KL}" ); fi
if [ -n "$BEST_CAIRL_INV" ];    then COMMON+=( "--grid" "irl.inv_coeff=${BEST_CAIRL_INV}" ); fi
if [ -n "$BEST_CAIRL_LATENT" ]; then COMMON+=( "--grid" "irl.latent_dim=${BEST_CAIRL_LATENT}" ); fi

$PY "${COMMON[@]}" | tee -a $LOG

echo "========== Causal-AIRL Experiments Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/causal_airl_scenarios.log"
