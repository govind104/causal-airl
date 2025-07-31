#!/bin/bash
#$ -N cartpole_comparison
#$ -q gpu
#$ -l gpu=1
#$ -pe sharedmem 2
#$ -l h_vmem=32G
#$ -l h_rt=01:00:00
#$ -wd /exports/eddie/scratch/s2696869/causal-airl
#$ -o logs/cartpole_comparison.out
#$ -e logs/cartpole_comparison.err
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