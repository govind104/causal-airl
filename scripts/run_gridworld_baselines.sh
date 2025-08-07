#!/bin/bash
#$ -N gridworld_baselines
#$ -q gpu
#$ -l gpu=1
#$ -pe sharedmem 2
#$ -l h_vmem=32G
#$ -l h_rt=01:00:00
#$ -wd /exports/eddie/scratch/s2696869/causal-airl
#$ -o logs/gridworld_baselines.out
#$ -e logs/gridworld_baselines.err
#$ -m beas
#$ -M s2696869@sms.ed.ac.uk
set -euo pipefail

# Load environment
. /etc/profile.d/modules.sh
module load anaconda
conda activate /exports/eddie/scratch/s2696869/.conda/envs/causal-irl-env

# Install in editable mode
pip install -e /exports/eddie/scratch/s2696869/causal-airl

# Ensure logs directory exists
mkdir -p logs

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
python -m visualisation.plot_scm_diagram | tee -a $LOG

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
