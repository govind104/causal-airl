#!/bin/bash
set -euo pipefail

# Setup log file
mkdir -p results/logs
LOG=results/logs/gridworld_baseline_maxent.log
echo "========== [MaxEnt GridWorld Baselines] ==========" > $LOG

# Config validation
echo "========== [0] Validating Configs ==========" | tee -a $LOG
python scripts/validate_config.py configs/gridworld_baseline_maxent.yaml | tee -a $LOG

# SCM Diagrams
echo "========== [1] Generating SCM Diagrams ==========" | tee -a $LOG
python -m visualisation.scm_diagram | tee -a $LOG

# Core experiment
echo "========== [2] Running MaxEnt Experiment ==========" | tee -a $LOG
python -m experiments.run_experiment --config configs/gridworld_baseline_maxent.yaml | tee -a $LOG

# Footer
echo "========== MaxEnt GridWorld Baselines Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/gridworld_baseline_maxent.log"