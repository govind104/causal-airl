#!/bin/bash
#$ -N generate_figures
#$ -pe sharedmem 2
#$ -l h_vmem=8G
#$ -l h_rt=00:30:00
#$ -wd /exports/eddie/scratch/s2696869/causal-airl
#$ -o logs/generate_figures.out
#$ -e logs/generate_figures.err
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
mkdir -p results/logs results/figures
LOG=results/logs/generate_figures.log
echo "========== [Generate Figures] ==========" > $LOG

echo "========== [1] SCM Diagrams ==========" | tee -a $LOG
python -m visualisation.plot_scm_diagram \
    --save_path "results/figures/plot_scm_diagram.png" | tee -a $LOG

echo "========== [2] GridWorld Reward Heatmaps ==========" | tee -a $LOG
python -m visualisation.plot_rewards \
    --reward_path "results/latest/gridworld_baselines/learned_reward.npy" \
    --terminals "[[4,4]]" \
    --save_path "results/figures/reward_heatmap.png" | tee -a $LOG

echo "========== [3] GridWorld Trajectory Rollouts ==========" | tee -a $LOG
python -m visualisation.plot_rollouts \
    --traj_dir "results/latest/gridworld_baselines" \
    --grid_size "(5,5)" \
    --terminals "[(4,4)]" \
    --save_path "results/figures/trajectory_rollouts.png" | tee -a $LOG

echo "========== [4] Policy Field Plots ==========" | tee -a $LOG
python -m visualisation.plot_policy_fields \
    --policy_path "results/latest/gridworld_baselines/policy.pt" \
    --grid_size "(5,5)" \
    --save_path "results/figures/policy_vector.png" | tee -a $LOG

echo "========== [5a] Training Curves ==========" | tee -a $LOG
python -m visualisation.plot_training_curves \
    --log_dir "results/airl_ablation/" \
    --save_path "results/figures/training_curves.pdf" | tee -a $LOG

echo "========== [5b] Ablation Summaries ==========" | tee -a $LOG
# Demos vs Reward Corr
python -m visualisation.plot_ablation_summaries \
    --results_dir "results/airl_ablation/" \
    --xparam "n_trajectories" \
    --metric "reward_corr" \
    --groupby "method" \
    --save_path "results/figures/ablation_reward_corr_vs_demos.pdf" | tee -a $LOG

# Gamma vs Value Difference
python -m visualisation.plot_ablation_summaries \
    --results_dir "results/airl_ablation/" \
    --xparam "gamma" \
    --metric "value_difference" \
    --groupby "method" \
    --save_path "results/figures/ablation_value_diff_vs_gamma.pdf" | tee -a $LOG

# Reward Type vs Policy Agreement
python -m visualisation.plot_ablation_summaries \
    --results_dir "results/airl_ablation/" \
    --xparam "reward_type" \
    --metric "policy_agreement" \
    --groupby "method" \
    --save_path "results/figures/ablation_policy_agreement_vs_reward_type.pdf" | tee -a $LOG

# Slip Probability vs Reward Corr
python -m visualisation.plot_ablation_summaries \
    --results_dir "results/airl_ablation/" \
    --xparam "slip_prob" \
    --metric "reward_corr" \
    --groupby "method" \
    --save_path "results/figures/ablation_reward_corr_vs_slip.pdf" | tee -a $LOG
    
echo "========== [6] Attribution Maps ==========" | tee -a $LOG
python -m visualisation.plot_attribution_maps \
    --run_dir "results/cartpole/latest" \
    --save_path "results/figures/cartpole_attributions.pdf" | tee -a $LOG

echo "========== [7] CartPole Reward Slices ==========" | tee -a $LOG
python -m visualisation.plot_cartpole_reward_curves \
    --run_dir "results/cartpole/latest" \
    --dims "0,2" \
    --save_path "results/figures/cartpole_reward_map.pdf" | tee -a $LOG

echo "========== [8] Reward Invariance (Variance Plots) ==========" | tee -a $LOG
python -m visualisation.plot_causal_invariance \
    --run_dir "results/latest/confounded/" \
    --save_path "results/figures/reward_invariance.pdf" | tee -a $LOG

echo "========== [9] Generalisation Performance ==========" | tee -a $LOG
python -m visualisation.plot_generalisation_results \
    --results_dir "results/generalization/" \
    --save_path "results/figures/generalisation_results.pdf" | tee -a $LOG

echo "========== [10] Summary Table ==========" | tee -a $LOG
python -m visualisation.generate_summary_tables \
    --results_dir "results/airl_ablation/" \
    --groupby "method,n_trajectories" \
    --metrics "reward_corr,policy_agreement,value_difference" \
    --out "results/figures/summary_table.tex" \
    --bold_best | tee -a $LOG

echo "========== Figure Generation Complete ==========" | tee -a $LOG
echo "Saved figures to results/figures/" | tee -a $LOG
