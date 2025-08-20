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

# Ensure logs directory exists
mkdir -p logs

# Setup log file
mkdir -p results/logs
LOG=results/logs/gridworld_baselines.log
echo "========== [GridWorld Baselines] ==========" > $LOG

# SCM Diagrams
echo "========== [1] Generating SCM Diagrams ==========" | tee -a $LOG
$PY -m visualisation.plot_scm_diagram | tee -a $LOG

# Core experiments
echo "========== [2] Running Experiments ==========" | tee -a $LOG

for seed in 42 123 456 789 2025; do
    for method in ng maxent airl causal_airl; do
        echo "=== Running $method baseline" | tee -a $LOG
        $PY -m experiments.run_experiment \
            --config "configs/gridworld_baseline_${method}.yaml" \
            --override "train.seed=${seed}" | tee -a $LOG
    done
done

# Footer
echo "========== GridWorld Baselines Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/gridworld_baselines.log"