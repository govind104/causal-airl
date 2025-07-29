#!/bin/bash
set -euo pipefail

# Setup log file
mkdir -p results/logs
LOG=results/logs/gridworld_baseline_ng.log
echo "========== [Ng-Russell GridWorld Baselines] ==========" > $LOG

# Config validation
echo "========== [0] Validating Config ==========" | tee -a $LOG
python scripts/validate_config.py configs/gridworld_baseline_ng.yaml | tee -a $LOG

# SCM Diagrams
echo "========== [1] Generating SCM Diagrams ==========" | tee -a $LOG
python -m visualisation.scm_diagram | tee -a $LOG

# Core experiment
echo "========== [2] Running Ng-Russell Experiment ==========" | tee -a $LOG
python -m experiments.run_experiment --config configs/gridworld_baseline_ng.yaml | tee -a $LOG

# Footer
echo "========== Ng-Russell GridWorld Baselines Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/gridworld_baseline_ng.log"