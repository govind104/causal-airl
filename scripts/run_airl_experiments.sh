#!/bin/bash
#$ -N airl_experiments
#$ -q gpu
#$ -l gpu=1
#$ -pe sharedmem 2
#$ -l h_vmem=32G
#$ -l h_rt=12:00:00
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

# Run parameter sweep
echo "========== [1] Running Parameter Sweep ==========" | tee -a $LOG

python -m experiments.sweeps \
  --base configs/airl_ablation.yaml \
  --save_root results/airl_ablation \
  --grid irl.gamma=0.9,0.95,0.99 \
  --grid expert.num_trajectories=5,10,20,50 \
  --grid env.slip_prob=0.0,0.1,0.2 \
  --grid env.reward_type=sparse,shaped | tee -a $LOG

# Optional enhanced sweep
# --sweep_params irl.gamma:expert.num_trajectories:env.slip_prob:env.reward_type:irl.entropy_coef:irl.grad_clip_norm
# --values "0.9,0.95,0.99:5,10,20,50:0.0,0.1,0.2:sparse,shaped:0.0,0.005,0.01:0.3,0.5,1.0"

# Footer
echo "========== AIRL Experiments Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/airl_experiments.log"
