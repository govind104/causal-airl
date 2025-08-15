#!/bin/bash
#$ -N cartpole_comparison
#$ -q gpu
#$ -l gpu=1
#$ -pe sharedmem 2
#$ -l h_vmem=32G
#$ -l h_rt=24:00:00
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
    for seed in 42 123 456; do
        for ntraj in 10 25 50; do
            for pole_len in 0.5 1.0; do
                echo "=== Running $method: seed=$seed traj=$ntraj pole_length=$pole_len ==="
                python -m experiments.run_experiment \
                --config "configs/cartpole_${method}.yaml" \
                --override "train.seed=${seed}" \
                --override "expert.num_trajectories=${ntraj}" \
                --override "expert.confounder_value=${pole_len}" | tee -a $LOG
            done
        done
    done
done

# Cross-confounder generalization
for method in airl causal_airl; do
    for seed in 42 123 456; do
        # Train on pole_length=0.5, test on pole_length=1.0
        python -m experiments.run_experiment \
            --config "configs/cartpole_${method}.yaml" \
            --override "train.seed=${seed}" \
            --override "expert.confounder_value=0.5" \
            --override "eval.test_z=1.0" | tee -a $LOG
    done
done

# Footer
echo "========== CartPole Comparison Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/cartpole_comparison.log"
