import os
import torch
import json
import argparse
import itertools
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from irl.airl import AIRLDiscriminator, create_cartpole_encoder
from irl.causal_airl import CausalAIRLDiscriminator


def plot_cartpole_reward_maps(reward_dict, dims=(0, 2), resolution=100, state_bounds=None, save_path=None, title_prefix=""):
    """
    Plot 2D reward maps over selected CartPole state dimensions.
    dims: tuple of 2 ints ∈ [0,1,2,3] — e.g., (0,2) = x vs theta
    reward_dict: key → 2D reward map numpy array
    """
    state_bounds = state_bounds or [[-2.4, 2.4], [-2, 2], [-0.42, 0.42], [-3.5, 3.5]]
    xdim, ydim = dims
    x = np.linspace(state_bounds[xdim][0], state_bounds[xdim][1], resolution)
    y = np.linspace(state_bounds[ydim][0], state_bounds[ydim][1], resolution)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, len(reward_dict), figsize=(6 * len(reward_dict), 5))
    if len(reward_dict) == 1:
        axes = [axes]

    for ax, (name, reward_map) in zip(axes, reward_dict.items()):
        if reward_map.shape != X.shape:
            raise ValueError(f"Reward map shape {reward_map.shape} does not match grid {X.shape}")
        im = ax.contourf(X, Y, reward_map, cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.set_title(f"{title_prefix} {name}")
        ax.set_xlabel(f"State dim {xdim}")
        ax.set_ylabel(f"State dim {ydim}")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Saved] {save_path}")
    plt.close()


def generate_state_grid(dims=(0, 2), resolution=100, bounds=None):
    bounds = bounds or [[-2.4, 2.4], [-2, 2], [-0.42, 0.42], [-3.5, 3.5]]
    xdim, ydim = dims
    x = np.linspace(bounds[xdim][0], bounds[xdim][1], resolution)
    y = np.linspace(bounds[ydim][0], bounds[ydim][1], resolution)
    X, Y = np.meshgrid(x, y)
    grid = np.zeros((resolution * resolution, 4), dtype=np.float32)
    for i, (xi, yi) in enumerate(itertools.product(x, y)):
        grid[i, xdim] = xi
        grid[i, ydim] = yi
    return X, Y, torch.FloatTensor(grid)

def compute_reward_map(run_dir, dims=(0, 2), resolution=100):
    model_path = os.path.join(run_dir, "model_weights.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model_weights.pt in {run_dir}")
    checkpoint = torch.load(model_path, map_location="cpu")
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)

    results = {}
    X, Y, state_grid = generate_state_grid(dims, resolution)

    method = cfg["irl"]["method"]
    irl_cfg = cfg.get("irl", {})
    gamma = irl_cfg.get("gamma", 0.99)
    latent_dim = irl_cfg.get("latent_dim", 2)
    inv_penalty = irl_cfg.get("invariance_penalty", 0.0)
    state_dim = cfg.get("state_dim", 4)
    action_dim = cfg.get("action_dim", 2)

    state_encoder = create_cartpole_encoder()
    if method == "causal-airl":
        discriminator = CausalAIRLDiscriminator(
            state_dim=state_dim, action_dim=action_dim, gamma=gamma, latent_dim=latent_dim, 
            invariance_penalty=inv_penalty, state_encoder=state_encoder
        )
    else:
        discriminator = AIRLDiscriminator(
            state_dim=state_dim, action_dim=action_dim, gamma=gamma, state_encoder=state_encoder
        )

    discriminator.load_state_dict(checkpoint["discriminator"])
    discriminator.eval()

    with torch.no_grad():
        rewards = discriminator.compute_reward(state_grid).reshape(resolution, resolution)
        results[method] = rewards.numpy()
    return results, X, Y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot AIRL/CausalAIRL reward heatmaps for CartPole")
    parser.add_argument("--run_dir", type=str, required=True, help="Experiment output directory")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save figure")
    parser.add_argument("--dims", type=str, default="0,2", help="Which state dims to slice on (comma-separated, e.g. 0,2)")
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    try:
        dims = tuple(int(i) for i in args.dims.split(","))
        assert len(dims) == 2 and all(0 <= d < 4 for d in dims)
    except:
        raise ValueError("Invalid --dims argument. Must be two comma-separated integers in [0,1,2,3], e.g. 0,2")

    reward_maps, X, Y = compute_reward_map(args.run_dir, dims=dims)
    plot_cartpole_reward_maps(reward_maps, dims=dims, save_path=args.save_path, title_prefix=args.title)
