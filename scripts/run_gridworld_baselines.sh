#!/bin/bash
set -euo pipefail

# Setup log file
mkdir -p results/logs
LOG=results/logs/gridworld_baselines.log
echo "========== [GridWorld Baselines] ==========" > $LOG

# Config validation
echo "========== [0] Validating Configs ==========" | tee -a $LOG
for config in configs/gridworld_baseline_*.yaml; do
    python scripts/validate_config.py "$config" | tee -a $LOG
done

# SCM Diagrams
echo "========== [1] Generating SCM Diagrams ==========" | tee -a $LOG
python -m visualisation.scm_diagram | tee -a $LOG

# Core experiments
echo "========== [2] Running Experiments ==========" | tee -a $LOG
for method in ng maxent airl causal_airl; do
    echo "=== Running $method baseline" | tee -a $LOG
    python -m experiments.run_experiment \
        --config "configs/gridworld_baseline_${method}.yaml" | tee -a $LOG
done

# Footer
echo "========== GridWorld Baselines Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/gridworld_baselines.log"