import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import ast
import glob

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
        fname = os.path.basename(path).replace('.npy', '')
        
        for traj in trajs:
            states = [step[0] for step in traj]
            xs = [s[1] for s in states]
            ys = [s[0] for s in states]
            plt.plot(xs, ys, '-o', color=color, alpha=0.7, label=fname)
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
    parser = argparse.ArgumentParser(description='Plot 2D trajectory rollouts')
    parser.add_argument('--traj_dir', type=str, required=True, help='Directory containing trajectory files')
    parser.add_argument('--grid_size', type=str, required=True, help='Grid size as tuple (rows, cols)')
    parser.add_argument('--terminals', type=str, required=True, help='Terminal states as list of tuples')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save output plot')
    
    args = parser.parse_args()
    
    # Parse inputs
    grid_size = ast.literal_eval(args.grid_size)
    terminals = ast.literal_eval(args.terminals)
    
    # Find all trajectory files
    pattern = os.path.join(args.traj_dir, 'trajectories*.npy')
    traj_files = glob.glob(pattern)
    
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files found in {args.traj_dir}")
    
    plot_trajectories(traj_files, grid_size, terminals, args.save_path)