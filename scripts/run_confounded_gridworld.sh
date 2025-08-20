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
: "${BEST_AIRL_ENTROPY:=}"
: "${BEST_AIRL_CLIP:=}"
: "${BEST_CAIRL_KL:=}"
: "${BEST_CAIRL_INV:=}"
: "${BEST_CAIRL_LATENT:=}"

# Setup log file
mkdir -p results/logs
LOG=results/logs/confounded_gridworld.log
echo "========== [Confounded GridWorld] ==========" > $LOG

# Core confounded experiment
echo "========== [1] Running Multi-Seed & Trajectory Experiments ==========" | tee -a $LOG

for method in airl causal_airl; do
  for z in 0 1; do
    for seed in 42 123 456 789 2025; do
      for ntraj in 10 20 40; do
        echo "=== Running $method with z=$z seed=$seed traj=$ntraj" | tee -a $LOG
        ARGS=( "--config" "configs/confounded_${method}_z${z}.yaml"
               "--override" "train.seed=${seed}"
               "--override" "expert.num_trajectories=${ntraj}"
               "--override" "eval.test_z=${z}"
               "--override" "eval.save_per_z=true" )
        if [ "$method" = "airl" ]; then
          [ -n "$BEST_AIRL_ENTROPY" ] && ARGS+=( "--override" "irl.entropy_coef=${BEST_AIRL_ENTROPY}" )
          [ -n "$BEST_AIRL_CLIP" ]    && ARGS+=( "--override" "irl.grad_clip_norm=${BEST_AIRL_CLIP}" )
        else
          [ -n "$BEST_AIRL_ENTROPY" ] && ARGS+=( "--override" "irl.entropy_coef=${BEST_AIRL_ENTROPY}" )
          [ -n "$BEST_AIRL_CLIP" ]    && ARGS+=( "--override" "irl.grad_clip_norm=${BEST_AIRL_CLIP}" )
          [ -n "$BEST_CAIRL_KL" ]     && ARGS+=( "--override" "irl.kl_coeff=${BEST_CAIRL_KL}" )
          [ -n "$BEST_CAIRL_INV" ]    && ARGS+=( "--override" "irl.inv_coeff=${BEST_CAIRL_INV}" )
          [ -n "$BEST_CAIRL_LATENT" ] && ARGS+=( "--override" "irl.latent_dim=${BEST_CAIRL_LATENT}" )
        fi
        $PY -m experiments.run_experiment "${ARGS[@]}" | tee -a $LOG
      done
    done
  done
done

echo "========== Confounded GridWorld Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/confounded_gridworld.log"
