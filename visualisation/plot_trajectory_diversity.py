import argparse
import os
import json
import math
import csv
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from visualisation.utils_config import find_run_dirs, flatten_config, get
from visualisation.style import setup_thesis_style, save_figure
from visualisation.scenario import label_scenario


def plot_trajectory_diversity(roots, out_dir):
    """Plot trajectory diversity metrics."""
    setup_thesis_style()
    run_dirs = find_run_dirs(roots)

    os.makedirs(out_dir, exist_ok=True)
    print(f"[diversity] roots={roots} | out={out_dir} | discovered_runs={len(run_dirs)}")

    # Collect data
    data = []

    for run_dir in run_dirs:
        metrics_path = os.path.join(run_dir, 'metrics.json')
        config_path = os.path.join(run_dir, 'config.json')
        env_path = os.path.join(run_dir, 'env_data.json')
        traj_path = os.path.join(run_dir, 'trajectories.npy')

        if not os.path.isfile(metrics_path) or not os.path.isfile(config_path):
            print(f"Skipping {run_dir} (missing files)")
            continue

        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            with open(config_path) as f:
                config = json.load(f)
            cfg_flat = flatten_config(config)
        except Exception as e:
            print(f"Skipping {run_dir} (parse error: {e})")
            continue

        # Pull metrics
        traj_entropy = metrics.get('trajectory_entropy')
        traj_overlap = metrics.get('trajectory_overlap')

        # Optional eval per-episode rates passthrough (may be absent)
        eval_success_rate = metrics.get('eval_success_rate', None)
        eval_timeout_rate = metrics.get('eval_timeout_rate', None)

        def _last_scalar(x):
            if x is None: return None
            if isinstance(x, (list, tuple)) and len(x) > 0:
                x = x[-1]
            try: return float(x)
            except Exception: return None

        method = get(cfg_flat, 'irl.method', 'unknown')
        scenario = label_scenario(cfg_flat)

        traj_entropy_val = _last_scalar(traj_entropy)
        traj_overlap_val = _last_scalar(traj_overlap)

        if traj_entropy_val is None or (isinstance(traj_entropy_val, float) and (math.isnan(traj_entropy_val))):
            continue

        # Defaults for trajectory-derived fields (filled if trajectories present)
        path_length_mean = float('nan')
        coverage = float('nan')
        unique_states = float('nan')
        n_episodes = float('nan')

        # Try to compute trajectory-derived metrics
        if os.path.isfile(traj_path):
            try:
                arr = np.load(traj_path, allow_pickle=True)
                lengths = []
                states = set()

                def safe_int(x):
                    try:
                        # tolerate numpy scalars / floats that are actually integers
                        return int(float(x))
                    except Exception:
                        return None

                def is_pair(obj):
                    # exactly two atomic (non-sequence) elements
                    if isinstance(obj, (list, tuple, np.ndarray)) and len(obj) == 2:
                        a, b = obj[0], obj[1]
                        if not isinstance(a, (list, tuple, np.ndarray)) and not isinstance(b, (list, tuple, np.ndarray)):
                            return True
                    return False

                def iter_states(obj):
                    """
                    Recursively yield (i, j) integer pairs from nested containers.
                    Accepts dense arrays, lists/tuples of pairs, or ragged mixtures.
                    """
                    if obj is None:
                        return
                    if isinstance(obj, np.ndarray):
                        if obj.ndim == 2 and obj.shape[-1] == 2 and obj.dtype != object:
                            for i, j in obj:
                                ii, jj = safe_int(i), safe_int(j)
                                if ii is not None and jj is not None:
                                    yield ii, jj
                            return
                        # fall through for object arrays or other ndims
                        for el in obj:
                            yield from iter_states(el)
                        return
                    if is_pair(obj):
                        ii, jj = safe_int(obj[0]), safe_int(obj[1])
                        if ii is not None and jj is not None:
                            yield ii, jj
                        return
                    if isinstance(obj, (list, tuple)):
                        for el in obj:
                            yield from iter_states(el)
                        return
                    # otherwise ignore (unknown atom)
                    return

                def consume_trajectory(tr):
                    # Collect states for a single trajectory and return its length
                    t_states = list(iter_states(tr))
                    for (ii, jj) in t_states:
                        states.add((ii, jj))
                    return len(t_states)

                # Fast-path dense format: [n_traj, T, 2]
                if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] == 2 and arr.dtype != object:
                    for tr in arr:
                        lengths.append(consume_trajectory(tr))
                else:
                    # Object/ragged or list-of-trajectories
                    # Treat `arr` as an iterable of trajectories when possible
                    if isinstance(arr, (list, tuple)) or (isinstance(arr, np.ndarray) and arr.dtype == object):
                        for tr in arr:
                            lengths.append(consume_trajectory(tr))
                    else:
                        # Single trajectory stored at top-level
                        lengths.append(consume_trajectory(arr))

                if lengths:
                    path_length_mean = float(np.mean(lengths))
                unique_states = float(len(states)) if states else float('nan')
                n_episodes = float(len(lengths)) if lengths else float('nan')

                # coverage requires grid size
                if os.path.isfile(env_path):
                    try:
                        with open(env_path, "r") as ef:
                            env = json.load(ef)
                        grid_size = env.get("grid_size", None)
                        if grid_size and len(grid_size) == 2 and all(x is not None for x in grid_size):
                            H, W = int(grid_size[0]), int(grid_size[1])
                            total = max(1, H * W)
                            if not math.isnan(unique_states):
                                coverage = float(unique_states / total)
                    except Exception as e:
                        print(f"[warn] {run_dir}: failed to read env_data.json for coverage: {e}")

                # concise log
                n_traj = len(lengths) if lengths else 0
                if n_traj > 0:
                    msg = f"len_mean={path_length_mean:.2f}, n_traj={n_traj}"
                else:
                    msg = "no trajectories parsed"
                cov_str = f"{coverage:.3f}" if not math.isnan(coverage) else "NaN"
                us_str = f"{int(unique_states)}" if not math.isnan(unique_states) else "NaN"
                ne_str = f"{int(n_traj)}" if n_traj > 0 else "NaN"
                print(f"[traj] {os.path.basename(run_dir)} | lengths={msg} unique_states={us_str} coverage={cov_str} n_episodes={ne_str}")
            except Exception as e:
                print(f"[warn] {run_dir}: failed to parse trajectories.npy: {e}")
                # leave NaNs

        record = {
            'method': method,
            'scenario': scenario,
            'trajectory_entropy': traj_entropy_val,
            'trajectory_overlap': traj_overlap_val,
            'run_dir': run_dir,
            'path_length_mean': path_length_mean,
            'coverage': coverage,
            'unique_states': unique_states,
            'n_episodes': n_episodes,
            'eval_success_rate': eval_success_rate,
            'eval_timeout_rate': eval_timeout_rate,
        }

        data.append(record)

    if not data:
        print("No trajectory diversity data found.")
        return

    # Sidecar CSV with per-run metrics (trajectory-derived included, NaNs if unavailable)
    sidecar_csv = os.path.join(out_dir, 'diversity_metrics.csv')
    try:
        with open(sidecar_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "run_dir", "scenario", "method",
                "trajectory_entropy", "trajectory_overlap",
                "path_length_mean", "coverage", "unique_states",
                "n_episodes", "eval_success_rate", "eval_timeout_rate",
            ])
            writer.writeheader()
            for r in data:
                writer.writerow({
                    "run_dir": r["run_dir"],
                    "scenario": r["scenario"],
                    "method": r["method"],
                    "trajectory_entropy": r["trajectory_entropy"],
                    "trajectory_overlap": r["trajectory_overlap"],
                    "path_length_mean": r["path_length_mean"],
                    "coverage": r["coverage"],
                    "unique_states": r["unique_states"],
                    "n_episodes": r["n_episodes"],
                    "eval_success_rate": r["eval_success_rate"],
                    "eval_timeout_rate": r["eval_timeout_rate"],
                })
        print(f"[diversity] wrote {sidecar_csv} (rows={len(data)})")
    except Exception as e:
        print(f"[warn] failed to write sidecar CSV: {e}")

    # Create plots
    scenarios = sorted(set(d['scenario'] for d in data))
    methods = sorted(set(d['method'] for d in data))

    # Check if we have overlap data
    has_overlap = any(d['trajectory_overlap'] is not None for d in data)

    n_metrics = 2 if has_overlap else 1
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    # Plot trajectory entropy
    ax = axes[0]
    scenario_method_data = {}

    for scenario in scenarios:
        scenario_method_data[scenario] = {}
        for method in methods:
            values = [d['trajectory_entropy'] for d in data
                     if d['scenario'] == scenario and d['method'] == method
                     and d['trajectory_entropy'] is not None]
            if values:
                scenario_method_data[scenario][method] = values

    # Create grouped boxplot for entropy
    x_pos = 0
    x_ticks = []
    x_labels = []

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    method_colors = {method: colors[i % len(colors)] for i, method in enumerate(methods)}

    for scenario in scenarios:
        if scenario not in scenario_method_data:
            continue

        scenario_methods = [m for m in methods if m in scenario_method_data[scenario]]
        n_methods = len(scenario_methods)

        if n_methods == 0:
            continue

        positions = [x_pos + i for i in range(n_methods)]
        data_for_box = [scenario_method_data[scenario][m] for m in scenario_methods]

        bp = ax.boxplot(data_for_box, positions=positions, patch_artist=True,
                        widths=0.6, tick_labels=scenario_methods)

        for patch, method in zip(bp['boxes'], scenario_methods):
            patch.set_facecolor(method_colors[method])
            patch.set_alpha(0.7)

        x_ticks.append(x_pos + n_methods/2 - 0.5)
        x_labels.append(scenario)
        x_pos += n_methods + 1

    ax.set_title('Trajectory Entropy by Scenario × Method')
    ax.set_ylabel('Trajectory Entropy')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    # Add legend
    legend_handles = [plt.Rectangle((0,0),1,1, facecolor=method_colors[m], alpha=0.7) for m in methods]
    ax.legend(legend_handles, methods, loc='upper right')

    # Plot trajectory overlap if available
    if has_overlap:
        ax = axes[1]
        overlap_data = {}

        for scenario in scenarios:
            overlap_data[scenario] = {}
            for method in methods:
                values = [d['trajectory_overlap'] for d in data
                         if d['scenario'] == scenario and d['method'] == method
                         and d['trajectory_overlap'] is not None]
                if values:
                    overlap_data[scenario][method] = values

        x_pos = 0
        x_ticks = []
        x_labels = []

        for scenario in scenarios:
            if scenario not in overlap_data:
                continue

            scenario_methods = [m for m in methods if m in overlap_data[scenario]]
            n_methods = len(scenario_methods)

            if n_methods == 0:
                continue

            positions = [x_pos + i for i in range(n_methods)]
            data_for_box = [overlap_data[scenario][m] for m in scenario_methods]

            bp = ax.boxplot(data_for_box, positions=positions, patch_artist=True,
                            widths=0.6, tick_labels=scenario_methods)

            for patch, method in zip(bp['boxes'], scenario_methods):
                patch.set_facecolor(method_colors[method])
                patch.set_alpha(0.7)

            x_ticks.append(x_pos + n_methods/2 - 0.5)
            x_labels.append(scenario)
            x_pos += n_methods + 1

        ax.set_title('Trajectory Overlap by Scenario × Method')
        ax.set_ylabel('Trajectory Overlap')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

    out_path = os.path.join(out_dir, 'diversity_boxplots.png')
    save_figure(fig, out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', nargs='+', required=True, help='Root directories with runs')
    parser.add_argument('--out', required=True, help='Output directory')
    args = parser.parse_args()

    plot_trajectory_diversity(args.roots, args.out)

if __name__ == '__main__':
    main()
