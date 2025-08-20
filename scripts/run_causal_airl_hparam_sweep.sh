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
LOG=results/logs/causal_airl_hparams.log
echo "========== [Causal-AIRL Experiments] ==========" > $LOG

# Run Causal-AIRL METHOD hyperparam sweep (kl_coeff, inv_coeff, latent_dim)
echo "========== [1] Running Causal-AIRL Hyperparam Sweep ==========" | tee -a $LOG

$PY -m experiments.sweeps \
  --base configs/confounded_causal_airl_z0.yaml \
  --save_root results/causal_airl_hparams \
  --grid train.seed=42,123,456,789,2025 \
  --grid expert.confounder_value=0,1 \
  --grid irl.kl_coeff=0.001,0.003,0.01 \
  --grid irl.inv_coeff=0.0,0.02,0.05 \
  --grid irl.latent_dim=2,4,8 | tee -a $LOG

# Footer
echo "========== Causal-AIRL Experiments Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/causal_airl_hparams.log"
