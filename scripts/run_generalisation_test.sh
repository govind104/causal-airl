#!/bin/bash
#$ -N generalisation_test
#$ -q gpu
#$ -l gpu=1
#$ -pe sharedmem 2
#$ -l h_vmem=32G
#$ -l h_rt=12:00:00
#$ -wd /exports/eddie/scratch/s2696869/causal-airl
#$ -o logs/generalisation_test.out
#$ -e logs/generalisation_test.err
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
LOG=results/logs/generalisation_test.log
echo "========== [Generalization Test] ==========" > $LOG

# Core experiments
echo "========== [1] Multi-Seed Cross-Confounder Testing ===========" | tee -a $LOG

for method in airl causal_airl; do
    for seed in 42 123 456; do
        for ntraj in 10 20 40; do
            echo "=== Running $method with seed=$seed traj=$ntraj ===" | tee -a $LOG
            # Train z=0, Test z=1
            python -m experiments.run_experiment \
                --config "configs/generalization_${method}.yaml" \
                --override "train.seed=${seed}" \
                --override "expert.num_trajectories=${ntraj}" \
                --override "expert.confounder_value=0" \
                --override "eval.test_z=1" | tee -a $LOG

            # Train z=1, Test z=0
            python -m experiments.run_experiment \
                --config "configs/generalization_${method}.yaml" \
                --override "train.seed=${seed}" \
                --override "expert.num_trajectories=${ntraj}" \
                --override "expert.confounder_value=1" \
                --override "eval.test_z=0" | tee -a $LOG
        done
    done
done


echo "========== [2] Spatial Generalization Testing ===========" | tee -a $LOG

heldout_regions=("top_left" "bottom_right" "top_right" "bottom_left")

for region in "${heldout_regions[@]}"; do
    for method in airl causal_airl; do
        for seed in 42 123 456; do
            echo "=== Running $method spatial test: region=$region seed=$seed ===" | tee -a $LOG
            python -m experiments.run_experiment \
            --config "configs/generalization_${method}.yaml" \
            --override "train.seed=${seed}" \
            --override "eval.heldout_region=${region}" | tee -a $LOG

            echo "=== Running $method spatial+confounder: region=$region train_z=0 test_z=1 ===" | tee -a $LOG
            python -m experiments.run_experiment \
            --config "configs/generalization_${method}.yaml" \
            --override "train.seed=${seed}" \
            --override "expert.confounder_value=0" \
            --override "eval.test_z=1" \
            --override "eval.heldout_region=${region}" | tee -a $LOG
        done
    done
done

# Footer
echo "========== Generalization Test Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/generalisation_test.log"
