import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_value_function(path):
    v = np.load(path)
    if v.ndim == 1:
        return v.reshape(-1)
    return v

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

def plot_value_heatmap(values, shape, terminals, title, save_path):
    values = values.reshape(shape)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(values, cmap="plasma", origin="upper")
    plt.colorbar(im, label="Value")
    plt.title(title)
    for (i, j) in terminals:
        plt.text(j, i, '★', ha='center', va='center', color='white', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[Saved] {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot value function heatmap from .npy file")
    parser.add_argument("--value_path", type=str, required=True, help="Path to V.npy or value_function.npy")
    parser.add_argument("--env_data", type=str, required=True, help="Path to env_data.json for grid info")
    parser.add_argument("--save_path", type=str, required=True, help="Output path for saved plot")
    parser.add_argument("--title", type=str, default="Value Function")
    args = parser.parse_args()

    V = load_value_function(args.value_path)
    shape = load_env_shape(args.env_data)
    terminals = load_terminal_states(args.env_data)
    plot_value_heatmap(V, shape, terminals, args.title, args.save_path)