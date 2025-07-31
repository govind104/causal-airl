#!/bin/bash
#$ -N confounded_gridworld
#$ -q gpu
#$ -l gpu=1
#$ -pe sharedmem 2
#$ -l h_vmem=32G
#$ -l h_rt=01:00:00
#$ -wd /exports/eddie/scratch/s2696869/causal-airl
#$ -o logs/confounded_gridworld.out
#$ -e logs/confounded_gridworld.err
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
        python -m experiments.run_experiment \
            --config "configs/confounded_${method}_z${z}.yaml" | tee -a $LOG
    done
done

# Footer
echo "========== Confounded GridWorld Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/confounded_gridworld.log"