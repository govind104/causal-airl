import os
import json
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_and_reshape_reward(path: str, env_data_path: str):
    reward = np.load(path)
    if reward.ndim == 1:
        with open(env_data_path, "r") as f:
            env_data = json.load(f)
        grid_size = env_data.get("grid_size") or env_data.get("grid_shape")
        if not grid_size:
            raise ValueError("Grid size missing in env_data.json")
        reward = reward.reshape(grid_size)
    return reward


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
    parser = argparse.ArgumentParser(description="Plot all reward maps in a run directory")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to experiment run directory")
    parser.add_argument("--save_prefix", type=str, default=None, help="Prefix for saved plots (no extension)")
    parser.add_argument("--title_prefix", type=str, default=None)
    parser.add_argument("--terminals", type=str, default=None)
    parser.add_argument("--arrows", action="store_true")
    args = parser.parse_args()

    env_data_path = os.path.join(args.run_dir, "env_data.json")
    reward_files = sorted(glob.glob(os.path.join(args.run_dir, "*reward*.npy")))
    if not reward_files:
        raise FileNotFoundError(f"No reward files found in {args.run_dir}")

    # Load terminal states if present
    if args.terminals:
        terminal_states = eval(args.terminals)
    else:
        with open(env_data_path, "r") as f:
            env_data = json.load(f)
        terminal_states = env_data.get("terminal_states", None)

    for reward_path in reward_files:
        reward_name = os.path.splitext(os.path.basename(reward_path))[0]
        reward = load_and_reshape_reward(reward_path, env_data_path)

        title = f"{args.title_prefix + ' - ' if args.title_prefix else ''}{reward_name.replace('_', ' ').title()}"
        save_name = f"{args.save_prefix}_{reward_name}.pdf" if args.save_prefix else None
        plot_reward_map(reward, title, save_name, terminal_states, args.arrows)