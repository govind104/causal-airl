#!/bin/bash
set -euo pipefail

# Setup log file
mkdir -p results/logs
LOG=results/logs/cartpole_comparison.log
echo "========== [CartPole Comparison] ==========" > $LOG

# Config validation
echo "========== [0] Validating Configs ==========" | tee -a $LOG
for config in configs/cartpole_*.yaml; do
    python scripts/validate_config.py "$config" | tee -a $LOG
done

# Core experiments
echo "========== [1] Running Experiments ==========" | tee -a $LOG
for method in airl causal_airl; do
    echo "=== Running $method on CartPole" | tee -a $LOG
    python -m experiments.run_experiment \
        --config "configs/cartpole_${method}.yaml" | tee -a $LOG
done

# Footer
echo "========== CartPole Comparison Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/cartpole_comparison.log"