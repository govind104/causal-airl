#!/usr/bin/env bash
# run_validation_suite.sh
# Reruns the 24 validation experiments, summarizes results, then renders all figures.

set -u  # continue on failures; we want the whole suite to try to run
ROOTS=("results/validation")
mkdir -p "${ROOTS[@]}"

run() {
  echo -e "\n>>> $*\n"
  "$@" || echo "✗ FAILED: $*"
}

# --- 24 experiments (order as in your table) ---

# 1 AIRL + Baseline GridWorld
run python -m experiments.run_experiment --config configs/gridworld_tiny_airl.yaml \
  --override expert.num_trajectories=20 --override env.reward_type=sparse \
  --override eval.save_dir=results/validation/airl_baseline

# 2 Causal-AIRL + Baseline GridWorld
run python -m experiments.run_experiment --config configs/gridworld_tiny_causal_airl.yaml \
  --override expert.num_trajectories=20 --override env.reward_type=sparse \
  --override eval.save_dir=results/validation/causal_airl_baseline

# 3 AIRL + Confounded Environment
run python -m experiments.run_experiment --config configs/gridworld_tiny_airl.yaml \
  --override env.name=ConfoundedGridWorld --override env.confounder_value=1 \
  --override expert.confounder_value=1 --override env.reward_type=sparse \
  --override eval.save_dir=results/validation/airl_confounded

# 4 Causal-AIRL + Confounded Environment
run python -m experiments.run_experiment --config configs/gridworld_tiny_causal_airl.yaml \
  --override env.name=ConfoundedGridWorld --override env.confounder_value=1 \
  --override expert.confounder_value=1 --override env.reward_type=sparse \
  --override eval.save_dir=results/validation/causal_airl_confounded

# 5 AIRL + Held-out Generalisation
run python -m experiments.run_experiment --config configs/gridworld_tiny_airl.yaml \
  --override eval.heldout_region=top_right --override expert.num_trajectories=30 \
  --override eval.save_dir=results/validation/airl_heldout

# 6 Causal-AIRL + Held-out Generalisation
run python -m experiments.run_experiment --config configs/gridworld_tiny_causal_airl.yaml \
  --override eval.heldout_region=top_right --override expert.num_trajectories=30 \
  --override eval.save_dir=results/validation/causal_airl_heldout

# 7 AIRL + Sparse Rewards
run python -m experiments.run_experiment --config configs/gridworld_tiny_airl.yaml \
  --override env.reward_type=sparse --override env.reward_value=1.0 \
  --override expert.num_trajectories=25 \
  --override eval.save_dir=results/validation/airl_sparse

# 8 Causal-AIRL + Sparse Rewards
run python -m experiments.run_experiment --config configs/gridworld_tiny_causal_airl.yaml \
  --override env.reward_type=sparse --override env.reward_value=1.0 \
  --override expert.num_trajectories=25 \
  --override eval.save_dir=results/validation/causal_airl_sparse

# 9 AIRL + Shaped Rewards
run python -m experiments.run_experiment --config configs/gridworld_tiny_airl.yaml \
  --override env.reward_type=shaped --override env.reward_value=1.0 \
  --override expert.num_trajectories=25 \
  --override eval.save_dir=results/validation/airl_shaped

# 10 Causal-AIRL + Shaped Rewards
run python -m experiments.run_experiment --config configs/gridworld_tiny_causal_airl.yaml \
  --override env.reward_type=shaped --override env.reward_value=1.0 \
  --override expert.num_trajectories=25 \
  --override eval.save_dir=results/validation/causal_airl_shaped

# 11 AIRL + Transition Noise
run python -m experiments.run_experiment --config configs/gridworld_tiny_airl.yaml \
  --override env.slip_prob=0.1 --override expert.num_trajectories=30 \
  --override eval.save_dir=results/validation/airl_noisy

# 12 Causal-AIRL + Transition Noise
run python -m experiments.run_experiment --config configs/gridworld_tiny_causal_airl.yaml \
  --override env.slip_prob=0.1 --override expert.num_trajectories=30 \
  --override eval.save_dir=results/validation/causal_airl_noisy

# 13 AIRL + Few-shot Learning
run python -m experiments.run_experiment --config configs/gridworld_tiny_airl.yaml \
  --override expert.num_trajectories=5 --override irl.max_iters=40 \
  --override eval.save_dir=results/validation/airl_fewshot

# 14 Causal-AIRL + Few-shot Learning
run python -m experiments.run_experiment --config configs/gridworld_tiny_causal_airl.yaml \
  --override expert.num_trajectories=5 --override irl.max_iters=40 \
  --override eval.save_dir=results/validation/causal_airl_fewshot

# 15 AIRL + Parameter Sensitivity (Gamma)
run python -m experiments.run_experiment --config configs/gridworld_tiny_airl.yaml \
  --override irl.gamma=0.8 --override expert.num_trajectories=20 \
  --override eval.save_dir=results/validation/airl_gamma_sweep

# 16 Causal-AIRL + Parameter Sensitivity (Gamma)
run python -m experiments.run_experiment --config configs/gridworld_tiny_causal_airl.yaml \
  --override irl.gamma=0.8 --override expert.num_trajectories=20 \
  --override eval.save_dir=results/validation/causal_airl_gamma_sweep

# 17 AIRL + Per-Z Invariance Evaluation
run python -m experiments.run_experiment --config configs/gridworld_tiny_airl.yaml \
  --override env.name=ConfoundedGridWorld --override env.confounder_value=1 \
  --override expert.confounder_value=1 --override irl.num_z_samples=5 \
  --override eval.save_per_z=true \
  --override eval.save_dir=results/validation/airl_perz

# 18 Causal-AIRL + Per-Z Invariance Evaluation
run python -m experiments.run_experiment --config configs/gridworld_tiny_causal_airl.yaml \
  --override env.name=ConfoundedGridWorld --override env.confounder_value=1 \
  --override expert.confounder_value=1 --override irl.num_z_samples=5 \
  --override eval.save_per_z=true \
  --override eval.save_dir=results/validation/causal_airl_perz

# 19 AIRL + Confounded (train z=0 → test z=1)
run python -m experiments.run_experiment --config configs/gridworld_tiny_airl.yaml \
  --override env.name=ConfoundedGridWorld --override env.confounder_value=0 \
  --override expert.confounder_value=0 --override eval.test_z=1 \
  --override env.reward_type=sparse --override expert.num_trajectories=20 \
  --override eval.save_dir=results/validation/airl_confounded_trainz0_testz1

# 20 Causal-AIRL + Confounded (train z=0 → test z=1)
run python -m experiments.run_experiment --config configs/gridworld_tiny_causal_airl.yaml \
  --override env.name=ConfoundedGridWorld --override env.confounder_value=0 \
  --override expert.confounder_value=0 --override eval.test_z=1 \
  --override env.reward_type=sparse --override expert.num_trajectories=20 \
  --override eval.save_dir=results/validation/causal_airl_confounded_trainz0_testz1

# 21 AIRL + Confounded (train z=1 → test z=0)
run python -m experiments.run_experiment --config configs/gridworld_tiny_airl.yaml \
  --override env.name=ConfoundedGridWorld --override env.confounder_value=1 \
  --override expert.confounder_value=1 --override eval.test_z=0 \
  --override env.reward_type=sparse --override expert.num_trajectories=20 \
  --override eval.save_dir=results/validation/airl_confounded_trainz1_testz0

# 22 Causal-AIRL + Confounded (train z=1 → test z=0)
run python -m experiments.run_experiment --config configs/gridworld_tiny_causal_airl.yaml \
  --override env.name=ConfoundedGridWorld --override env.confounder_value=1 \
  --override expert.confounder_value=1 --override eval.test_z=0 \
  --override env.reward_type=sparse --override expert.num_trajectories=20 \
  --override eval.save_dir=results/validation/causal_airl_confounded_trainz1_testz0

# 23 AIRL + Gamma=0.95
run python -m experiments.run_experiment --config configs/gridworld_tiny_airl.yaml \
  --override irl.gamma=0.95 --override expert.num_trajectories=20 \
  --override eval.save_dir=results/validation/airl_gamma_095

# 24 Causal-AIRL + Gamma=0.95
run python -m experiments.run_experiment --config configs/gridworld_tiny_causal_airl.yaml \
  --override irl.gamma=0.95 --override expert.num_trajectories=20 \
  --override eval.save_dir=results/validation/causal_airl_gamma_095

# --- build tables/figures after experiments ---

run python -m experiments.summarize_results --roots results/validation

# All plots/tables (orchestrator):
run python -m visualisation.make_all --roots results/validation --out results/figures
