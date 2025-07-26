import os
from visualisation.plot_rewards import plot_reward_map
from visualisation.plot_policies import plot_policy_vector_field
from visualisation.plot_metrics import plot_metric_vs_param
from visualisation.plot_training_curves import plot_training_curve
from visualisation.plot_cartpole_rewards import plot_reward_over_state_space
from visualisation.generate_summary_tables import generate_table, save_table
from visualisation.scm_diagram import plot_gridworld_scm, plot_causal_airl_scm
import numpy as np
import pandas as pd
import json


def make_all_figures():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("tables", exist_ok=True)

    # --- Reward Heatmaps (AIRL, MaxEnt, Ng) ---
    reward = np.load("results/gridworld/airl/reward.npy")
    plot_reward_map(reward, title="AIRL Reward", save_path="figures/airl_reward.png", terminal_states=[(4,4)])

    reward = np.load("results/gridworld/maxent/reward.npy")
    plot_reward_map(reward, title="MaxEnt Reward", save_path="figures/maxent_reward.png", terminal_states=[(4,4)])

    # --- Policy Vector Fields ---
    policy = np.load("results/gridworld/airl/policy.npy")
    plot_policy_vector_field(policy, grid_shape=(5,5), title="AIRL Policy", save_path="figures/airl_policy.png", terminals=[(4,4)])

    # --- Training Curves ---
    with open("results/gridworld/airl/train_log_airl.json", "r") as f:
        log = json.load(f)
    x = list(range(len(log["disc_loss"])))
    plot_training_curve(x, {"disc_loss": log["disc_loss"]}, title="AIRL Discriminator Loss", save_path="figures/airl_loss.png")

    # --- CartPole Reward Slice ---
    from models.reward_network import RewardNet
    import torch
    reward_net = RewardNet(input_dim=6)
    reward_net.load_state_dict(torch.load("results/cartpole/airl/reward.pt", map_location="cpu"))
    plot_reward_over_state_space(reward_fn=reward_net, dims=(0,2), save_path="figures/cartpole_reward.png")

    # --- Metric Sweep Plots ---
    df = pd.read_csv("results/gridworld_sweeps/summary.csv")
    plot_metric_vs_param(df, metric="reward_corr", xparam="demos", hue="method", save_path="figures/reward_corr_vs_demos.png", title="Reward Correlation vs Demos")
    plot_metric_vs_param(df, metric="policy_agreement", xparam="demos", hue="method", save_path="figures/policy_agreement_vs_demos.png")

    # --- Tables ---
    table = generate_table(df, groupby=["method", "demos"], metrics=["reward_corr", "policy_agreement"])
    save_table(table, "tables/summary_reward_policy.tex", fmt="latex")

    # --- SCM Diagrams ---
    plot_gridworld_scm(save_path="figures/scm_gridworld.png")
    plot_causal_airl_scm(save_path="figures/scm_causal_airl.png")


if __name__ == "__main__":
    make_all_figures()
