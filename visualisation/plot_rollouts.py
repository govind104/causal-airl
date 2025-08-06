import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectories(traj_paths, grid_size, terminals, save_path):
    plt.figure(figsize=(10, 8))
    
    # Draw grid
    for i in range(grid_size[0] + 1):
        plt.axhline(y=i, color='k', linestyle='-', alpha=0.1)
    for j in range(grid_size[1] + 1):
        plt.axvline(x=j, color='k', linestyle='-', alpha=0.1)
    
    # Plot terminals
    for term in terminals:
        plt.plot(term[1], term[0], 'r*', markersize=15, zorder=5)
    
    # Plot trajectories
    for path in traj_paths:
        trajs = np.load(path, allow_pickle=True)
        color = np.random.rand(3,)
        label_base = os.path.basename(path).replace('.npy', '')
        
        for traj in trajs:
            states = [step[0] for step in traj if step[0] is not None]
            xs = [s[1] for s in states]
            ys = [s[0] for s in states]
            plt.plot(xs, ys, '-o', color=color, alpha=0.7, label=label_base)
            plt.plot(xs[0], ys[0], 'go', markersize=8, alpha=0.7)  # Start state
    
    # Add legend and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), loc='best')
    
    plt.xlim(-0.5, grid_size[1] - 0.5)
    plt.ylim(-0.5, grid_size[0] - 0.5)
    plt.gca().invert_yaxis()  # Row 0 at top
    plt.title('Trajectory Rollouts')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot expert/learner trajectory rollouts from a run directory')
    parser.add_argument('--run_dir', type=str, required=True, help='Path to experiment run directory')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save output plot')
    parser.add_argument('--prefix', type=str, default="trajectories", help='Filename prefix to filter (e.g. trajectories or learner_trajectories)')
    
    args = parser.parse_args()
    
    # Load env metadata
    env_data_path = os.path.join(args.run_dir, "env_data.json")
    with open(env_data_path, "r") as f:
        env_data = json.load(f)
    grid_size = env_data.get("grid_size") or env_data.get("grid_shape")
    terminals = env_data.get("terminal_states", [])

    # Find relevant trajectory files
    pattern = os.path.join(args.run_dir, f"{args.prefix}*.npy")
    traj_files = glob.glob(pattern)
    
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found in {args.run_dir} matching prefix '{args.prefix}'")

    plot_trajectories(traj_files, grid_size, terminals, args.save_path)