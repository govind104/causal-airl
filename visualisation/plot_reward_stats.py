import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
# Try SciPy, fallback to NumPy implementation
try:
    from scipy import stats as _scipy_stats
except Exception:
    _scipy_stats = None

from visualisation.utils_config import find_run_dirs, flatten_config, get
from visualisation.style import setup_thesis_style, save_figure
from visualisation.scenario import label_scenario


def compute_reward_stats(reward_map):
    """Compute reward statistics from reward map."""
    rewards_flat = reward_map.flatten()
    rewards_abs = np.abs(rewards_flat)

    # Remove zeros for certain calculations
    nonzero_rewards = rewards_abs[rewards_abs > 1e-6]

    stats_dict = {
        'reward_sparsity': np.mean(rewards_abs < 1e-6),
        'reward_std': np.std(rewards_flat),
        'reward_skewness': 0.0,
    }

    if len(rewards_flat) > 2:
        if _scipy_stats is not None:
            try:
                stats_dict['reward_skewness'] = float(_scipy_stats.skew(rewards_flat))
            except Exception:
                pass
        if stats_dict['reward_skewness'] == 0.0:
            mu = float(np.mean(rewards_flat))
            sigma = float(np.std(rewards_flat, ddof=0))
            stats_dict['reward_skewness'] = float(np.mean((rewards_flat - mu) ** 3) / (sigma ** 3 + 1e-12)) if sigma > 1e-12 else 0.0

    # Gini coefficient for absolute values
    if len(nonzero_rewards) > 1:
        sorted_rewards = np.sort(nonzero_rewards)
        n = len(sorted_rewards)
        index = np.arange(1, n + 1)
        gini = 2 * np.sum(index * sorted_rewards) / (n * np.sum(sorted_rewards)) - (n + 1) / n
        stats_dict['reward_gini_abs'] = gini
    else:
        stats_dict['reward_gini_abs'] = 0

    # Histogram entropy for absolute values
    if len(nonzero_rewards) > 0:
        hist, _ = np.histogram(nonzero_rewards, bins=10, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        stats_dict['reward_hist_entropy_abs'] = entropy
    else:
        stats_dict['reward_hist_entropy_abs'] = 0

    return stats_dict

def plot_reward_stats(roots, out_dir):
    """Plot reward statistics."""
    setup_thesis_style()
    run_dirs = find_run_dirs(roots)

    # First-line diagnostics + deterministic jitter
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(0)
    print(f"[reward_stats] roots={roots} | out={out_dir} | discovered_runs={len(run_dirs)}")

    # Collect data
    data = []

    for run_dir in run_dirs:
        config_path = os.path.join(run_dir, 'config.json')
        if not os.path.isfile(config_path):
            print(f"Skipping {run_dir} (missing config.json)")
            continue

        try:
            with open(config_path) as f:
                config = json.load(f)
            cfg_flat = flatten_config(config)
        except Exception as e:
            print(f"Skipping {run_dir} (config parse error: {e})")
            continue

        method = get(cfg_flat, 'irl.method', 'unknown')
        scenario = label_scenario(cfg_flat)

        # Try to load reward_stats.json first
        stats_path = os.path.join(run_dir, 'reward_stats.json')
        if os.path.isfile(stats_path):
            try:
                with open(stats_path) as f:
                    reward_stats = json.load(f)
            except Exception as e:
                print(f"Skipping {run_dir} (reward_stats parse error: {e})")
                continue
        else:
            # Compute stats from reward map
            reward_map_paths = [
                os.path.join(run_dir, 'learned_reward_map.npy'),
                os.path.join(run_dir, 'learned_reward.npy')
            ]

            reward_map = None
            for path in reward_map_paths:
                if os.path.isfile(path):
                    try:
                        reward_map = np.load(path)
                        break
                    except Exception as e:
                        continue

            if reward_map is None:
                print(f"Skipping {run_dir} (no reward map found)")
                continue

            reward_stats = compute_reward_stats(reward_map)

        record = {
            'run_dir': run_dir,
            'method': method,
            'scenario': scenario,
            **reward_stats
        }
        data.append(record)

    if not data:
        print("No reward stats data found.")
        return

    # Create aggregated comparison plots by scenario
    scenarios = sorted(set(d['scenario'] for d in data))
    stat_keys = ['reward_sparsity', 'reward_gini_abs', 'reward_hist_entropy_abs', 'reward_std', 'reward_skewness']

    for scenario in scenarios:
        scenario_data = [d for d in data if d['scenario'] == scenario]
        methods = sorted(set(d['method'] for d in scenario_data))

        fig, axes = plt.subplots(1, len(stat_keys), figsize=(4*len(stat_keys), 4))
        if len(stat_keys) == 1:
            axes = [axes]

        for i, stat_key in enumerate(stat_keys):
            ax = axes[i]

            method_values = {}
            for method in methods:
                values = [d[stat_key] for d in scenario_data if d['method'] == method and stat_key in d]
                if values:
                    method_values[method] = values

            # Log per-stat sample counts
            if method_values:
                counts_log = {m: len(vs) for m, vs in method_values.items()}
                print(f"[reward_stats] {scenario} | {stat_key}: {counts_log}")

            if method_values:
                positions = list(range(len(method_values)))
                bp = ax.boxplot([method_values[m] for m in method_values.keys()],
                                positions=positions, patch_artist=True, tick_labels=list(method_values.keys()))

                # Color boxes
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # Swarm overlay (jittered scatter) — same color mapping as boxes
                method_list = list(method_values.keys())
                for k, method in enumerate(method_list):
                    vals = method_values[method]
                    color = colors[k % len(colors)]
                    # Jitter each point in bucket k
                    for val in vals:
                        jitter = np.random.uniform(-0.12, 0.12)
                        ax.scatter(positions[k] + jitter, val,
                                   s=16, alpha=0.45, color=color, zorder=3, linewidths=0)

            ax.set_title(stat_key.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle(f'Reward Statistics: {scenario}')
        out_path = os.path.join(out_dir, f'{scenario}_stats_by_method.png')
        save_figure(fig, out_path)
        print(f"[reward_stats] wrote {out_path}")

    # Create per-run panels for detailed view
    for record in data:
        run_id = os.path.basename(record['run_dir'])

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for i, stat_key in enumerate(stat_keys):
            ax = axes[i]
            value = record.get(stat_key, 0)
            ax.bar([stat_key.replace('_', ' ').title()], [value])
            ax.set_title(f"{stat_key.replace('_', ' ').title()}: {value:.4f}")
            ax.grid(True, alpha=0.3)

        # Hide extra subplot
        axes[-1].set_visible(False)

        plt.suptitle(f'Reward Stats: {record["method"]} - {record["scenario"]} - {run_id}')
        out_path = os.path.join(out_dir, f'{run_id}_stats.png')
        save_figure(fig, out_path)
        print(f"[reward_stats] wrote {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', nargs='+', required=True, help='Root directories with runs')
    parser.add_argument('--out', required=True, help='Output directory')
    args = parser.parse_args()

    plot_reward_stats(args.roots, args.out)

if __name__ == '__main__':
    main()
