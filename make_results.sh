#!/bin/bash
set -euo pipefail
mkdir -p results

echo "========== [1] SCM Diagrams =========="
python experiments/visualise_scm.py

echo "========== [2] Baseline IRL Methods on GridWorld =========="
for method in ng maxent airl causal_airl; do
  sed -i "s/method: .*/method: \"$method\"/" experiments/config_gridworld.yaml
  python experiments/run_experiment.py --config experiments/config_gridworld.yaml
done

echo "========== [3] Ablation Sweeps on GridWorld =========="
python experiments/sweeps.py

echo "========== [4] Stochasticity Sweep (Slippery GridWorld) =========="
for slip in 0.0 0.2; do
  sed -i "s/name: .*/name: \"SlipperyGridWorld\"/" experiments/config_gridworld.yaml
  sed -i "s/slip_prob: .*/slip_prob: $slip/" experiments/config_gridworld.yaml
  sed -i "s/method: .*/method: \"airl\"/" experiments/config_gridworld.yaml
  python experiments/run_experiment.py --config experiments/config_gridworld.yaml
done

echo "========== [5] Confounding Tests (Confounded GridWorld) =========="
sed -i "s/name: .*/name: \"ConfoundedGridWorld\"/" experiments/config_gridworld.yaml
sed -i "s/confounded: .*/confounded: true/" experiments/config_gridworld.yaml
sed -i "s/method: .*/method: \"causal_airl\"/" experiments/config_gridworld.yaml
python experiments/run_experiment.py --config experiments/config_gridworld.yaml

echo "========== [6] Generalisation Test (Z=0 training, Z=1 eval) =========="
sed -i "s/confounder_value: null/confounder_value: 0/" experiments/config_gridworld.yaml
python experiments/run_experiment.py --config experiments/config_gridworld.yaml

echo "========== [7] Continuous Control with CartPole =========="
sed -i "s/method: .*/method: \"causal_airl\"/" experiments/config_cartpole.yaml
python experiments/run_experiment.py --config experiments/config_cartpole.yaml

echo "========== [8] Generate Final Tables =========="
python visualisation/generate_summary_tables.py --input results/gridworld_sweeps/summary.csv --output results/tables/gridworld_summary.tex
python visualisation/generate_summary_tables.py --input results/cartpole/summary.csv --output results/tables/cartpole_summary.tex

echo "========== [9] Plot Training Curves =========="
python visualisation/plot_training_curves.py --log results/gridworld/train_log_airl.json --save results/figures/airl_curve.png
python visualisation/plot_training_curves.py --log results/gridworld/train_log_causal_airl.json --save results/figures/causal_airl_curve.png

echo "========== [10] Plot Metric Trends =========="
python visualisation/plot_metrics.py --input results/gridworld_sweeps/summary.csv --x demos --y reward_corr --hue method --save results/figures/ablation_reward_corr.png
python visualisation/plot_metrics.py --input results/gridworld_sweeps/summary.csv --x slip --y policy_agreement --hue method --save results/figures/ablation_policy_agreement.png

echo "========== [✔] ALL EXPERIMENTS COMPLETE =========="

