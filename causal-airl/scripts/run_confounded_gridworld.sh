#!/bin/bash
set -euo pipefail

# Setup log file
mkdir -p results/logs
LOG=results/logs/confounded_gridworld.log
echo "========== [Confounded GridWorld] ==========" > $LOG

# Config validation
echo "========== [0] Validating Configs ==========" | tee -a $LOG
for config in configs/confounded_*.yaml; do
    python scripts/validate_config.py "$config" | tee -a $LOG
done

# Core experiments
echo "========== [1] Running Experiments ==========" | tee -a $LOG
for method in airl causal_airl; do
    for z in 0 1; do
        echo "=== Running $method with z=$z" | tee -a $LOG
        python experiments/run_experiment.py \
            --config "configs/confounded_${method}_z${z}.yaml" | tee -a $LOG
    done
done

# Footer
echo "========== Confounded GridWorld Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/confounded_gridworld_log.txt"