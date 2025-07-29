#!/bin/bash
set -euo pipefail

# Setup log file
mkdir -p results/logs
LOG=results/logs/generalisation_test.log
echo "========== [Generalization Test] ==========" > $LOG

# Config validation
echo "========== [0] Validating Configs ==========" | tee -a $LOG
for config in configs/generalization_*.yaml; do
    python scripts/validate_config.py "$config" | tee -a $LOG
done

# Core experiments
echo "========== [1] Running Experiments ==========" | tee -a $LOG
for method in airl causal_airl; do
    echo "=== Running $method generalization" | tee -a $LOG
    python -m experiments.run_experiment \
        --config "configs/generalization_${method}.yaml" | tee -a $LOG
done

# Generate symlinks to latest results
echo "========== [2] Creating Symlinks ==========" | tee -a $LOG
bash scripts/generate_latest_symlinks.sh | tee -a $LOG

# Footer
echo "========== Generalization Test Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/generalisation_test.log"