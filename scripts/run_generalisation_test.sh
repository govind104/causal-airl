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

# Optional best hyperparams (set/export these before running; defaults = no override)
: "${BEST_AIRL_ENTROPY:=}"
: "${BEST_AIRL_CLIP:=}"
: "${BEST_CAIRL_KL:=}"
: "${BEST_CAIRL_INV:=}"
: "${BEST_CAIRL_LATENT:=}"

# Setup log file
mkdir -p results/logs
LOG=results/logs/generalisation_test.log
echo "========== [Generalization Test] ==========" > $LOG

# Core experiments
echo "========== [1] Multi-Seed Cross-Confounder Testing ===========" | tee -a $LOG

for method in airl causal_airl; do
    for seed in 42 123 456 789 2025; do
        for ntraj in 10 20 40; do
            for slip in 0.0 0.1; do
                echo "=== Running $method with seed=$seed traj=$ntraj ===" | tee -a $LOG
                # Train z=0, Test z=1
                ARGS=( "--config" "configs/generalization_${method}.yaml"
                      "--override" "train.seed=${seed}"
                      "--override" "expert.num_trajectories=${ntraj}"
                      "--override" "env.slip_prob=${slip}"
                      "--override" "expert.confounder_value=0"
                      "--override" "eval.test_z=1" )
                if [ "$method" = "airl" ]; then
                [ -n "$BEST_AIRL_ENTROPY" ] && ARGS+=( "--override" "irl.entropy_coef=${BEST_AIRL_ENTROPY}" )
                [ -n "$BEST_AIRL_CLIP" ]    && ARGS+=( "--override" "irl.grad_clip_norm=${BEST_AIRL_CLIP}" )
                else
                [ -n "$BEST_CAIRL_KL" ]     && ARGS+=( "--override" "irl.kl_coeff=${BEST_CAIRL_KL}" )
                [ -n "$BEST_CAIRL_INV" ]    && ARGS+=( "--override" "irl.inv_coeff=${BEST_CAIRL_INV}" )
                [ -n "$BEST_CAIRL_LATENT" ] && ARGS+=( "--override" "irl.latent_dim=${BEST_CAIRL_LATENT}" )
                fi
                $PY -m experiments.run_experiment "${ARGS[@]}" | tee -a $LOG

                # Train z=1, Test z=0
                ARGS=( "--config" "configs/generalization_${method}.yaml"
                      "--override" "train.seed=${seed}"
                      "--override" "expert.num_trajectories=${ntraj}"
                      "--override" "env.slip_prob=${slip}"
                      "--override" "expert.confounder_value=1"
                      "--override" "eval.test_z=0" )
                if [ "$method" = "airl" ]; then
                [ -n "$BEST_AIRL_ENTROPY" ] && ARGS+=( "--override" "irl.entropy_coef=${BEST_AIRL_ENTROPY}" )
                [ -n "$BEST_AIRL_CLIP" ]    && ARGS+=( "--override" "irl.grad_clip_norm=${BEST_AIRL_CLIP}" )
                else
                [ -n "$BEST_CAIRL_KL" ]     && ARGS+=( "--override" "irl.kl_coeff=${BEST_CAIRL_KL}" )
                [ -n "$BEST_CAIRL_INV" ]    && ARGS+=( "--override" "irl.inv_coeff=${BEST_CAIRL_INV}" )
                [ -n "$BEST_CAIRL_LATENT" ] && ARGS+=( "--override" "irl.latent_dim=${BEST_CAIRL_LATENT}" )
                fi
                $PY -m experiments.run_experiment "${ARGS[@]}" | tee -a $LOG
            done
        done
    done
done


echo "========== [2] Spatial Generalization Testing ===========" | tee -a $LOG

heldout_regions=("top_left" "bottom_right" "top_right" "bottom_left")

for region in "${heldout_regions[@]}"; do
    for method in airl causal_airl; do
        for seed in 42 123 456 789 2025; do
            echo "=== Running $method spatial test: region=$region seed=$seed ===" | tee -a $LOG
            ARGS=( "--config" "configs/generalization_${method}.yaml"
                   "--override" "train.seed=${seed}"
                   "--override" "eval.heldout_region=${region}" )
            if [ "$method" = "airl" ]; then
              [ -n "$BEST_AIRL_ENTROPY" ] && ARGS+=( "--override" "irl.entropy_coef=${BEST_AIRL_ENTROPY}" )
              [ -n "$BEST_AIRL_CLIP" ]    && ARGS+=( "--override" "irl.grad_clip_norm=${BEST_AIRL_CLIP}" )
            else
              [ -n "$BEST_CAIRL_KL" ]     && ARGS+=( "--override" "irl.kl_coeff=${BEST_CAIRL_KL}" )
              [ -n "$BEST_CAIRL_INV" ]    && ARGS+=( "--override" "irl.inv_coeff=${BEST_CAIRL_INV}" )
              [ -n "$BEST_CAIRL_LATENT" ] && ARGS+=( "--override" "irl.latent_dim=${BEST_CAIRL_LATENT}" )
            fi
            $PY -m experiments.run_experiment "${ARGS[@]}" | tee -a $LOG

            echo "=== Running $method spatial+confounder: region=$region train_z=0 test_z=1 ===" | tee -a $LOG
            ARGS=( "--config" "configs/generalization_${method}.yaml"
                   "--override" "train.seed=${seed}"
                   "--override" "expert.confounder_value=0"
                   "--override" "eval.test_z=1"
                   "--override" "eval.heldout_region=${region}" )
            if [ "$method" = "airl" ]; then
              [ -n "$BEST_AIRL_ENTROPY" ] && ARGS+=( "--override" "irl.entropy_coef=${BEST_AIRL_ENTROPY}" )
              [ -n "$BEST_AIRL_CLIP" ]    && ARGS+=( "--override" "irl.grad_clip_norm=${BEST_AIRL_CLIP}" )
            else
              [ -n "$BEST_CAIRL_KL" ]     && ARGS+=( "--override" "irl.kl_coeff=${BEST_CAIRL_KL}" )
              [ -n "$BEST_CAIRL_INV" ]    && ARGS+=( "--override" "irl.inv_coeff=${BEST_CAIRL_INV}" )
              [ -n "$BEST_CAIRL_LATENT" ] && ARGS+=( "--override" "irl.latent_dim=${BEST_CAIRL_LATENT}" )
            fi
            $PY -m experiments.run_experiment "${ARGS[@]}" | tee -a $LOG

            echo "=== Running $method spatial+confounder: region=$region train_z=1 test_z=0 ===" | tee -a $LOG
            ARGS=( "--config" "configs/generalization_${method}.yaml"
                   "--override" "train.seed=${seed}"
                   "--override" "expert.confounder_value=1"
                   "--override" "eval.test_z=0"
                   "--override" "eval.heldout_region=${region}" )
            if [ "$method" = "airl" ]; then
              [ -n "$BEST_AIRL_ENTROPY" ] && ARGS+=( "--override" "irl.entropy_coef=${BEST_AIRL_ENTROPY}" )
              [ -n "$BEST_AIRL_CLIP" ]    && ARGS+=( "--override" "irl.grad_clip_norm=${BEST_AIRL_CLIP}" )
            else
              [ -n "$BEST_CAIRL_KL" ]     && ARGS+=( "--override" "irl.kl_coeff=${BEST_CAIRL_KL}" )
              [ -n "$BEST_CAIRL_INV" ]    && ARGS+=( "--override" "irl.inv_coeff=${BEST_CAIRL_INV}" )
              [ -n "$BEST_CAIRL_LATENT" ] && ARGS+=( "--override" "irl.latent_dim=${BEST_CAIRL_LATENT}" )
            fi
            $PY -m experiments.run_experiment "${ARGS[@]}" | tee -a $LOG
        done
    done
done

# Footer
echo "========== Generalization Test Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/generalisation_test.log"