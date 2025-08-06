import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_reward(reward_path):
    reward = np.load(reward_path)
    if reward.ndim == 1:
        return reward.reshape(-1)
    return reward

def load_env_shape(env_data_path):
    with open(env_data_path, "r") as f:
        env_data = json.load(f)
    if "grid_size" in env_data:
        return tuple(env_data["grid_size"])
    else:
        raise ValueError("grid_size not found in env_data.json")

def load_terminal_states(env_data_path):
    with open(env_data_path, "r") as f:
        env_data = json.load(f)
    return env_data.get("terminal_states", [])

def compute_reward_difference(r1, r2, mode="abs"):
    if r1.shape != r2.shape:
        raise ValueError(f"Shape mismatch: {r1.shape} vs {r2.shape}")
    if mode == "abs":
        diff = np.abs(r1 - r2)
    elif mode == "signed":
        diff = r1 - r2
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return diff

def plot_diff_heatmap(diff, shape, terminals, title, save_path):
    diff = diff.reshape(shape)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(diff, cmap="bwr", origin="upper")
    plt.colorbar(im, label="Reward Difference")
    plt.title(title)
    for (i, j) in terminals:
        plt.text(j, i, '★', ha='center', va='center', color='black', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[Saved] {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot reward difference heatmap between two runs")
    parser.add_argument("--run_dir_1", type=str, required=True, help="Path to first run directory")
    parser.add_argument("--run_dir_2", type=str, required=True, help="Path to second run directory")
    parser.add_argument("--reward_key", type=str, default="learned_reward.npy",
                        help="Name of reward file (e.g. learned_reward.npy, causal_reward.npy)")
    parser.add_argument("--mode", type=str, default="abs", choices=["abs", "signed"],
                        help="Difference mode: absolute or signed")
    parser.add_argument("--save_path", type=str, required=True, help="Output path for saved plot")
    parser.add_argument("--title", type=str, default="Reward Difference")
    args = parser.parse_args()

    r1_path = os.path.join(args.run_dir_1, args.reward_key)
    r2_path = os.path.join(args.run_dir_2, args.reward_key)
    env1_path = os.path.join(args.run_dir_1, "env_data.json")
    terminals = load_terminal_states(env1_path)
    shape = load_env_shape(env1_path)

    r1 = load_reward(r1_path)
    r2 = load_reward(r2_path)
    diff = compute_reward_difference(r1, r2, mode=args.mode)

    plot_diff_heatmap(diff, shape, terminals, args.title, args.save_path)