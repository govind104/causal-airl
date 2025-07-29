#!/bin/bash
set -euo pipefail

# Setup log file
mkdir -p results/logs
LOG=results/logs/gridworld_baseline_airl.log
echo "========== [AIRL GridWorld Baselines] ==========" > $LOG

# Config validation
echo "========== [0] Validating Configs ==========" | tee -a $LOG
python scripts/validate_config.py configs/gridworld_baseline_airl.yaml | tee -a $LOG

# SCM Diagrams
echo "========== [1] Generating SCM Diagrams ==========" | tee -a $LOG
python -m visualisation.scm_diagram | tee -a $LOG

# Core experiment
echo "========== [2] Running AIRL Experiment ==========" | tee -a $LOG
python -m experiments.run_experiment --config configs/gridworld_baseline_airl.yaml | tee -a $LOG

# Footer
echo "========== AIRL GridWorld Baselines Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/gridworld_baseline_airl.log"