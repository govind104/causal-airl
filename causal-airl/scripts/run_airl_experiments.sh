#!/bin/bash
set -euo pipefail

# Setup log file
mkdir -p results/logs
LOG=results/logs/airl_experiments.log
echo "========== [AIRL Experiments] ==========" > $LOG

# Config validation
echo "========== [0] Validating Config ==========" | tee -a $LOG
python scripts/validate_config.py configs/airl_ablation.yaml | tee -a $LOG

# Run parameter sweep
echo "========== [1] Running Parameter Sweep ==========" | tee -a $LOG
python experiments/sweeps.py \
    --base_config configs/airl_ablation.yaml \
    --sweep_params irl.method:irl.gamma:expert.num_trajectories:env.slip_prob:env.reward_type \
    --values "airl,causal_airl:0.9,0.95,0.99:5,10,20,50:0.0,0.1,0.2:sparse,shaped" \
    --save_dir results/airl_ablation | tee -a $LOG

# Footer
echo "========== AIRL Experiments Complete ==========" | tee -a $LOG
echo "Saving log to results/logs/airl_experiments_log.txt"