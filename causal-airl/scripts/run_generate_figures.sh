#!/bin/bash
set -euo pipefail

# Setup log file
mkdir -p results/logs
LOG=results/logs/generate_figures.log
echo "========== [Generate Figures] ==========" > $LOG

# Generate all figures
echo "========== [1] Running All Visualizations ==========" | tee -a $LOG
python visualisation/scm_diagram.py \
    --save_path "results/figures/scm_diagram.png" | tee -a $LOG

python visualisation/plot_rewards.py \
    --reward_path "results/latest/gridworld_baselines/learned_reward.npy" \
    --terminals "[[4,4]]" \
    --save_path "results/figures/reward_heatmap.png" | tee -a $LOG

python visualisation/plot_rollouts.py \
    --traj_dir "results/latest/gridworld_baselines" \
    --grid_size "(5,5)" \
    --terminals "[(4,4)]" \
    --save_path "results/figures/trajectory_rollouts.png" | tee -a $LOG

python visualisation/plot_training_curves.py \
    --log_path "results/latest/airl_ablation/training_logs.json" \
    --save_path "results/figures/training_curves.png" | tee -a $LOG

python visualisation/plot_metrics.py \
    --csv "results/summary.csv" \
    --metric "reward_corr" \
    --xparam "demos" \
    --hue "method" \
    --save_path "results/figures/reward_vs_demos.png" | tee -a $LOG

python visualisation/plot_reward_invariance.py \
    --reward_dir "results/latest/confounded" \
    --save_path "results/figures/reward_variance.png" | tee -a $LOG

python visualisation/generate_summary_tables.py \
    --csv "results/summary.csv" \
    --groupby "method" \
    --metrics "reward_corr,policy_agreement" \
    --out "results/figures/summary_table.tex" | tee -a $LOG

python visualisation/plot_generalisation_performance.py \
    --csv "results/generalization_summary.csv" \
    --save_path "results/figures/generalization.png" | tee -a $LOG

python visualisation/plot_policies.py \
    --policy_path "results/latest/gridworld_baselines/policy.npy" \
    --grid_size "(5,5)" \
    --save_path "results/figures/policy_vector.png" | tee -a $LOG

python visualisation/plot_cartpole_rewards.py \
    --model_path "results/latest/cartpole/model_weights.pt" \
    --save_path "results/figures/cartpole_rewards.png" | tee -a $LOG

# Footer
echo "========== Figure Generation Complete ==========" | tee -a $LOG
echo "Saved figures to results/figures/" | tee -a $LOG