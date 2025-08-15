#!/bin/bash
#$ -N confounded_gridworld
#$ -q gpu
#$ -l gpu=1
#$ -pe sharedmem 2
#$ -l h_vmem=32G
#$ -l h_rt=12:00:00
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

# Core confounded experiments

# Use BEST parameters from your AIRL and Causal-AIRL sweeps
# BEST_AIRL_GAMMA="0.99"
# BEST_AIRL_LR="0.0003"
# BEST_AIRL_ENTROPY="0.005"
#
# BEST_CAUSAL_GAMMA="0.99"
# BEST_CAUSAL_LR="0.0003"
# BEST_CAUSAL_KL="0.003"
# BEST_CAUSAL_INV="0.1"
# BEST_CAUSAL_LATENT="4"

# for method in airl causal_airl; do
#   for z in 0 1; do
#     echo "=== Running $method with z=$z (optimized params)" | tee -a $LOG
#
#     if [ "$method" = "airl" ]; then
#       python -m experiments.run_experiment \
#         --config "configs/confounded_${method}_z${z}.yaml" \
#         --override "irl.gamma=$BEST_AIRL_GAMMA" \
#         --override "irl.lr=$BEST_AIRL_LR" \
#         --override "irl.entropy_coef=$BEST_AIRL_ENTROPY" \
#         --override "irl.max_iters=200" | tee -a $LOG
#     else
#       python -m experiments.run_experiment \
#         --config "configs/confounded_${method}_z${z}.yaml" \
#         --override "irl.gamma=$BEST_CAUSAL_GAMMA" \
#         --override "irl.lr=$BEST_CAUSAL_LR" \
#         --override "irl.kl_coeff=$BEST_CAUSAL_KL" \
#         --override "irl.invariance_penalty=$BEST_CAUSAL_INV" \
#         --override "irl.latent_dim=$BEST_CAUSAL_LATENT" \
#         --override "irl.max_iters=200" | tee -a $LOG
#     fi
#   done
# done

echo "========== [1] Running Multi-Seed & Trajectory Experiments ==========" | tee -a $LOG

for method in airl causal_airl; do
  for z in 0 1; do
    for seed in 42 123 456; do
      for ntraj in 10 20 40; do
        echo "=== Running $method with z=$z seed=$seed traj=$ntraj" | tee -a $LOG
        python -m experiments.run_experiment \
          --config "configs/confounded_${method}_z${z}.yaml" \
          --override "train.seed=${seed}" \
          --override "expert.num_trajectories=${ntraj}" | tee -a $LOG
      done
    done
  done
done

echo "========== Confounded GridWorld Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/confounded_gridworld.log"
