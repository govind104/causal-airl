import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse


def load_reward(path: str):
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".json"):
        import json
        with open(path, "r") as f:
            data = json.load(f)
        return np.array(data["reward"])
    else:
        raise ValueError(f"Unsupported format: {path}")


def plot_reward_map(
    reward: np.ndarray,
    title: str = "Reward Heatmap",
    save_path: str = None,
    terminal_states=None,
    arrow_overlay=False
):
    plt.figure(figsize=(5, 4))
    ax = plt.gca()

    im = ax.imshow(reward, cmap="coolwarm", origin="upper")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    if terminal_states:
        for (i, j) in terminal_states:
            plt.text(j, i, '★', ha='center', va='center', color='black', fontsize=14)

    if arrow_overlay:
        # optional vector field overlay (placeholder)
        dx = np.zeros_like(reward)
        dy = np.zeros_like(reward)
        ax.quiver(np.arange(reward.shape[1]), np.arange(reward.shape[0]), dx, dy, color='k')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[Saved] {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--title", type=str, default="Reward Heatmap")
    parser.add_argument("--terminals", type=str, default=None)
    parser.add_argument("--arrows", action="store_true")
    args = parser.parse_args()

    reward = load_reward(args.reward_path)
    terminal_states = eval(args.terminals) if args.terminals else None
    plot_reward_map(reward, args.title, args.save_path, terminal_states, args.arrows)
