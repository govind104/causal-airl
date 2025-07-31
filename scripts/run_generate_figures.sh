#!/bin/bash
#$ -N generate_figures
#$ -q gpu
#$ -l gpu=1
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
mkdir -p results/logs
LOG=results/logs/generate_figures.log
echo "========== [Generate Figures] ==========" > $LOG

# Generate all figures
echo "========== [1] Running All Visualizations ==========" | tee -a $LOG
python -m visualisation.scm_diagram \
    --save_path "results/figures/scm_diagram.png" | tee -a $LOG

python -m visualisation.plot_rewards \
    --reward_path "results/latest/gridworld_baselines/learned_reward.npy" \
    --terminals "[[4,4]]" \
    --save_path "results/figures/reward_heatmap.png" | tee -a $LOG

python -m visualisation.plot_rollouts \
    --traj_dir "results/latest/gridworld_baselines" \
    --grid_size "(5,5)" \
    --terminals "[(4,4)]" \
    --save_path "results/figures/trajectory_rollouts.png" | tee -a $LOG

python -m visualisation.plot_training_curves \
    --log_path "results/latest/airl_ablation/training_logs.json" \
    --save_path "results/figures/training_curves.png" | tee -a $LOG

python -m visualisation.plot_metrics \
    --csv "results/summary.csv" \
    --metric "reward_corr" \
    --xparam "demos" \
    --hue "method" \
    --save_path "results/figures/reward_vs_demos.png" | tee -a $LOG

python -m visualisation.plot_reward_invariance \
    --reward_dir "results/latest/confounded" \
    --save_path "results/figures/reward_variance.png" | tee -a $LOG

python -m visualisation.generate_summary_tables \
    --csv "results/summary.csv" \
    --groupby "method" \
    --metrics "reward_corr,policy_agreement" \
    --out "results/figures/summary_table.tex" | tee -a $LOG

python -m visualisation.plot_generalisation_performance \
    --csv "results/generalization_summary.csv" \
    --save_path "results/figures/generalization.png" | tee -a $LOG

python -m visualisation.plot_policies \
    --policy_path "results/latest/gridworld_baselines/policy.npy" \
    --grid_size "(5,5)" \
    --save_path "results/figures/policy_vector.png" | tee -a $LOG

python -m visualisation.plot_cartpole_rewards \
    --model_path "results/latest/cartpole/model_weights.pt" \
    --save_path "results/figures/cartpole_rewards.png" | tee -a $LOG

# Footer
echo "========== Figure Generation Complete ==========" | tee -a $LOG
echo "Saved figures to results/figures/" | tee -a $LOG