import argparse
import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import glob

from matplotlib.gridspec import GridSpec

from visualisation.utils_config import find_run_dirs, flatten_config, get
from visualisation.style import setup_thesis_style, save_figure
from visualisation.scenario import label_scenario


def plot_checkpoint_evolution(roots, out_dir):
    """Plot reward evolution across checkpoints."""
    setup_thesis_style()
    run_dirs = find_run_dirs(roots)

    # First-line diagnostics
    print(f"[ckpt] roots={roots} | out={out_dir} | discovered_runs={len(run_dirs)}")
    os.makedirs(out_dir, exist_ok=True)

    for run_dir in run_dirs:
        # Check for checkpoints
        checkpoint_pattern = os.path.join(run_dir, 'checkpoints', 'reward_iter_*.npy')
        checkpoint_files = sorted(glob.glob(checkpoint_pattern))

        if not checkpoint_files:
            print(f"Skipping {run_dir} (no checkpoints found)")
            continue

        # Load environment data for grid shape
        env_data_path = os.path.join(run_dir, 'env_data.json')
        if not os.path.isfile(env_data_path):
            print(f"Skipping {run_dir} (missing env_data.json)")
            continue

        try:
            with open(env_data_path) as f:
                env_data = json.load(f)
            grid_size = env_data.get('grid_size', [4, 4])
        except Exception as e:
            print(f"Skipping {run_dir} (env_data parse error: {e})")
            continue

        # Load metrics for timeline
        metrics_path = os.path.join(run_dir, 'metrics.json')
        if os.path.isfile(metrics_path):
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                reward_corr = get(metrics, 'reward_correlation', [])
            except:
                reward_corr = []
        else:
            reward_corr = []

        # Select up to 5 checkpoints evenly spaced
        n_checkpoints = min(5, len(checkpoint_files))
        if n_checkpoints < 2:
            print(f"Skipping {run_dir} (too few checkpoints)")
            continue

        indices = np.linspace(0, len(checkpoint_files)-1, n_checkpoints, dtype=int)
        selected_files = [checkpoint_files[i] for i in indices]

        # Create figure
        fig = plt.figure(constrained_layout=True, figsize=(12, 8))
        gs = fig.add_gridspec(2, n_checkpoints, height_ratios=[3, 1], hspace=0.3, wspace=0.3)

        # Plot checkpoints
        reward_maps = []
        iterations = []
        # Sidecar rows: one per plotted checkpoint
        sidecar_rows = []

        for i, checkpoint_file in enumerate(selected_files):
            try:
                reward_map = np.load(checkpoint_file)
                if reward_map.ndim == 1:
                    reward_map = reward_map.reshape(grid_size)
                reward_maps.append(reward_map)

                # Extract iteration from filename
                iter_num = int(os.path.basename(checkpoint_file).split('_')[-1].split('.')[0])
                iterations.append(iter_num)

                ax = fig.add_subplot(gs[0, i])
                im = ax.imshow(reward_map, cmap='RdBu_r', origin='upper')
                ax.set_title(f'Iter {iter_num}')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, shrink=0.8)

                # Compute simple stats for sidecar
                r_min = float(np.min(reward_map))
                r_max = float(np.max(reward_map))
                r_mean = float(np.mean(reward_map))
                sidecar_rows.append({
                    "iteration": int(iter_num),
                    "r_min": r_min,
                    "r_max": r_max,
                    "r_mean": r_mean,
                    # reward_correlation_at_iter filled below after we validate the series
                    "reward_correlation_at_iter": None,
                })

            except Exception as e:
                print(f"Error loading {checkpoint_file}: {e}")
                continue

        # Guard against scalar or too-short series; keep a validated ref
        rc_series = reward_corr if isinstance(reward_corr, (list, tuple)) and len(reward_corr) > 1 else None
        # Fill sidecar reward_correlation_at_iter if available
        if rc_series is not None:
            for row in sidecar_rows:
                it = row["iteration"]
                row["reward_correlation_at_iter"] = float(rc_series[it]) if 0 <= it < len(rc_series) else float("nan")
        else:
            for row in sidecar_rows:
                row["reward_correlation_at_iter"] = float("nan")

        # Plot metric timeline if available
        if rc_series is not None:
            ax_timeline = fig.add_subplot(gs[1, :])
            x_vals = list(range(len(rc_series)))
            ax_timeline.plot(x_vals, rc_series, 'o-', alpha=0.7)

            # Mark checkpoint positions
            for iter_num in iterations:
                if iter_num < len(rc_series):
                    ax_timeline.axvline(iter_num, color='red', linestyle='--', alpha=0.5)
                    ax_timeline.plot(iter_num, rc_series[iter_num], 'ro', markersize=8)

            ax_timeline.set_xlabel('Iteration')
            ax_timeline.set_ylabel('Reward Correlation')
            ax_timeline.grid(True, alpha=0.3)

        # Generate run ID from directory name
        run_id = os.path.basename(run_dir)
        out_path = os.path.join(out_dir, f'{run_id}_evolution.png')
        save_figure(fig, out_path, tight=False)

        # Write sidecar CSV matching plotted checkpoints
        try:
            sidecar_csv = os.path.join(out_dir, f"{run_id}_evolution.csv")
            with open(sidecar_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "iteration", "r_min", "r_max", "r_mean", "reward_correlation_at_iter"
                ])
                writer.writeheader()
                for row in sidecar_rows:
                    writer.writerow(row)
            print(f"[sidecar] wrote {sidecar_csv} (rows={len(sidecar_rows)})")
        except Exception as e:
            print(f"[warn] failed to write sidecar CSV for {run_id}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', nargs='+', required=True, help='Root directories with runs')
    parser.add_argument('--out', required=True, help='Output directory')
    args = parser.parse_args()

    plot_checkpoint_evolution(args.roots, args.out)

if __name__ == '__main__':
    main()
