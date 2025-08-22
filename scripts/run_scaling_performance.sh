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

# Setup artifacts
mkdir -p results/logs results/scaling
LOG=results/logs/scaling_performance.log
CSV=results/scaling/perf.csv
echo "method,size,seed,wall_clock_s" > "$CSV"
echo "========== [Scaling Study: Grid Size vs Performance] ==========" > $LOG

sizes=(5 7 9 11)
seeds=(42 123 456 789 2025)

for size in "${sizes[@]}"; do
  # Keep demo density roughly stable with size (tweak if needed)
  if   [ "$size" -le 6 ]; then NTRAJ=20
  elif [ "$size" -le 8 ]; then NTRAJ=30
  else                           NTRAJ=40
  fi
  for seed in "${seeds[@]}"; do
    for method in ng maxent airl causal_airl; do
      echo "=== $method size=${size}x${size} seed=$seed ntraj=$NTRAJ ===" | tee -a $LOG
      if [ "$method" = "airl" ]; then
        BASE="configs/airl_ablation.yaml"
        OVRS=( "irl.entropy_coef=${BEST_AIRL_ENTROPY}" "irl.grad_clip_norm=${BEST_AIRL_CLIP}"
               "eval.save_dir=results/scaling/airl" )
      elif [ "$method" = "causal_airl" ]; then
        BASE="configs/causal_airl_ablation.yaml"
        OVRS=( "irl.entropy_coef=${BEST_AIRL_ENTROPY}" "irl.grad_clip_norm=${BEST_AIRL_CLIP}"
               "irl.kl_coeff=${BEST_CAIRL_KL}" "irl.inv_coeff=${BEST_CAIRL_INV}" "irl.latent_dim=${BEST_CAIRL_LATENT}"
               "irl.kl_warmup_epochs=100" "eval.save_dir=results/scaling/causal_airl" )
      elif [ "$method" = "ng" ]; then
        BASE="configs/gridworld_baseline_ng.yaml"
        OVRS=( "eval.save_dir=results/scaling/ng" )
      else
        BASE="configs/gridworld_baseline_maxent.yaml"
        OVRS=( "eval.save_dir=results/scaling/maxent" )
      fi
      START=$(date +%s)
      $PY -m experiments.run_experiment \
        --config "$BASE" \
        --override "train.seed=${seed}" \
        --override "expert.num_trajectories=${NTRAJ}" \
        --override "env.grid_size=[${size},${size}]" \
        --override "env.reward_type=sparse" \
        --override "env.slip_prob=0.0" \
        --override "irl.max_iters=200" \
        $(for o in "${OVRS[@]}"; do printf -- ' --override %q' "$o"; done) | tee -a $LOG
      END=$(date +%s)
      echo "$method,$size,$seed,$((END-START))" >> "$CSV"
    done
  done
done

echo "========== Scaling Study Complete ==========" | tee -a $LOG
echo "CSV: $CSV" | tee -a $LOG
