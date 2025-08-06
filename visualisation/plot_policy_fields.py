import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from models.policy import PolicyNet  # adapt if your model is elsewhere


ACTION_TO_DELTA = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}


def load_policy(run_dir, device="cpu"):
    policy_path = os.path.join(run_dir, "policy.pt")
    config_path = os.path.join(run_dir, "config.json")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    grid_size = cfg.get("grid_size", [5, 5])
    state_dim = int(np.prod(grid_size))
    action_dim = 4

    policy = PolicyNet(state_dim, action_dim)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    return policy, cfg


def plot_vector_field(policy, grid_size, save_path, terminal_states=None):
    W, H = grid_size
    X, Y = np.meshgrid(np.arange(H), np.arange(W))
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)

    for i in range(W):
        for j in range(H):
            idx = i * H + j
            state = torch.zeros(state_dim).float()
            state[idx] = 1.0  # one-hot
            with torch.no_grad():
                probs = policy(state.unsqueeze(0)).squeeze()
                action = torch.argmax(probs).item()

            dx, dy = ACTION_TO_DELTA.get(action, (0, 0))
            U[i, j] = dy
            V[i, j] = -dx  # invert to match plot orientation

    plt.figure(figsize=(6, 5))
    plt.quiver(X, Y, U, V, scale=1, scale_units='xy', angles='xy')
    plt.xlim(-0.5, H - 0.5)
    plt.ylim(-0.5, W - 0.5)
    plt.gca().invert_yaxis()
    plt.xticks(range(H))
    plt.yticks(range(W))
    plt.grid(True, alpha=0.2)

    if terminal_states:
        for (i, j) in terminal_states:
            plt.plot(j, i, 'r*', markersize=15)

    plt.title("Learned Policy Field")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[Saved] {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learned policy as vector field")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Experiment run directory with policy.pt and env_data.json")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Where to save the plot")
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, "env_data.json")) as f:
        env_data = json.load(f)

    grid_size = env_data.get("grid_size") or env_data.get("grid_shape", [5, 5])
    terminal_states = env_data.get("terminal_states", [])

    global state_dim
    state_dim = np.prod(grid_size)

    policy, _ = load_policy(args.run_dir)
    plot_vector_field(policy, grid_size, args.save_path, terminal_states)