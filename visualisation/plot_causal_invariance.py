import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

def compute_reward_variance(reward_files):
    """Compute pixelwise variance across reward maps"""
    reward_maps = []
    for file in reward_files:
        reward_maps.append(np.load(file))
    
    stack = np.stack(reward_maps, axis=0)
    variance = np.var(stack, axis=0)
    mean_variance = np.mean(variance)
    return variance, mean_variance

def plot_variance_heatmap(variance, title="Reward Variance", save_path=None):
    plt.figure(figsize=(6, 5))
    plt.imshow(variance, cmap="hot", origin="upper")
    plt.colorbar(label="Variance")
    plt.title(f"{title} (Mean: {np.mean(variance):.4f})")
    plt.axis("off")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved variance heatmap to {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot reward invariance heatmap")
    parser.add_argument("--reward_dir", type=str, required=True, help="Directory with reward_map_z*.npy files")
    parser.add_argument("--save_path", type=str, required=True, help="Output path for plot")
    parser.add_argument("--title", type=str, default="Reward Variance Across Z")
    args = parser.parse_args()
    
    reward_files = glob.glob(os.path.join(args.reward_dir, "reward_map_z*.npy"))
    variance, mean_var = compute_reward_variance(reward_files)
    plot_variance_heatmap(variance, args.title, args.save_path)
    print(f"Mean Variance: {mean_var:.4f}")