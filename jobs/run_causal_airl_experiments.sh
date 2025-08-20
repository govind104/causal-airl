#!/bin/bash
#$ -N causal_airl_experiments
#$ -q gpu
#$ -l gpu=1
#$ -pe sharedmem 2
#$ -l h_vmem=32G
#$ -l h_rt=12:00:00
#$ -wd /exports/eddie/scratch/s2696869/causal-airl
#$ -o logs/causal_airl_experiments.out
#$ -e logs/causal_airl_experiments.err
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
LOG=results/logs/causal_airl_experiments.log
echo "========== [Causal-AIRL Experiments] ==========" > $LOG

# Run parameter sweep
echo "========== [1] Running Parameter Sweep ==========" | tee -a $LOG

python -m experiments.sweeps \
  --base configs/causal_airl_ablation.yaml \
  --save_root results/causal_airl_ablation \
  --grid train.seed=42,123,456,789,2025 \
  --grid irl.kl_coeff=0.001,0.003,0.01 \
  --grid irl.invariance_penalty=0.05,0.1,0.2 \
  --grid irl.inv_coeff=0.0,0.02,0.05 \
  --grid irl.latent_dim=2,4,8 | tee -a $LOG

# Explicit no-penalty control for ablation clarity
echo "========== [2] No-Penalty Control Run ==========" | tee -a $LOG
python -m experiments.run_experiment \
  --config configs/causal_airl_ablation.yaml \
  --override "irl.inv_coeff=0.0" | tee -a $LOG

# Optional enhanced sweep
# --sweep_params irl.kl_coeff:irl.invariance_penalty:irl.inv_coeff:irl.latent_dim:irl.num_z_samples:irl.entropy_coef
# --values "0.001,0.003,0.01:0.05,0.1,0.2:0.0,0.02,0.05:2,4,8:3,5,10:0.0,0.005,0.01"

# Footer
echo "========== Causal-AIRL Experiments Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/causal_airl_experiments.log"
