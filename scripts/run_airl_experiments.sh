#!/bin/bash
#$ -N airl_experiments
#$ -q gpu
#$ -l gpu=1
#$ -pe sharedmem 2
#$ -l h_vmem=32G
#$ -l h_rt=22:30:00
#$ -wd /exports/eddie/scratch/s2696869/causal-airl
#$ -o logs/airl_experiments.out
#$ -e logs/airl_experiments.err
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
LOG=results/logs/airl_experiments.log
echo "========== [AIRL Experiments] ==========" > $LOG

# Config validation
echo "========== [0] Validating Config ==========" | tee -a $LOG
python scripts/validate_config.py configs/airl_ablation.yaml | tee -a $LOG

# Run parameter sweep
echo "========== [1] Running Parameter Sweep ==========" | tee -a $LOG
python -m experiments.sweeps \
    --base_config configs/airl_ablation.yaml \
    --sweep_params irl.method:irl.gamma:expert.num_trajectories:env.slip_prob:env.reward_type \
    --values "airl,causal_airl:0.9,0.95,0.99:5,10,20,50:0.0,0.1,0.2:sparse,shaped" \
    --save_dir results/airl_ablation | tee -a $LOG

# Footer
echo "========== AIRL Experiments Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/airl_experiments.log"
