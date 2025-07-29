import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def plot_reward_over_state_space(
    reward_fn,
    dims=(0, 2),  # default: x-position and pole angle
    state_bounds=[[-2.4, 2.4], [-2, 2], [-0.42, 0.42], [-3.5, 3.5]],
    a=0,
    resolution=100,
    title="Reward Landscape",
    save_path=None
):
    """Plot reward over a 2D slice of the CartPole state space."""
    xdim, ydim = dims
    x = np.linspace(state_bounds[xdim][0], state_bounds[xdim][1], resolution)
    y = np.linspace(state_bounds[ydim][0], state_bounds[ydim][1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            s = np.zeros(4)
            s[xdim] = X[i, j]
            s[ydim] = Y[i, j]
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            a_tensor = torch.tensor([a], dtype=torch.long)
            a_onehot = torch.nn.functional.one_hot(a_tensor, num_classes=2).float()
            sa = torch.cat([s_tensor, a_onehot], dim=1)
            r = reward_fn(sa)
            Z[i, j] = r.item()

    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, Z, cmap="viridis")
    plt.colorbar()
    plt.xlabel(f"State dim {xdim}")
    plt.ylabel(f"State dim {ydim}")
    plt.title(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[Saved] {save_path}")
    plt.close()
