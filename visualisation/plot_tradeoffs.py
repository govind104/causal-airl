import argparse
import os
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from typing import Optional, Any, Dict
from matplotlib.ticker import MaxNLocator

from visualisation.utils_config import flatten_config, find_run_dirs, get
from visualisation.style import setup_thesis_style, save_figure
from visualisation.scenario import label_scenario


def _last(x: Any):
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return x[-1]
    return x

def load_data_from_metrics(run_dirs):
    """Load final metrics from run directories."""
    records = []
    for run_dir in run_dirs:
        metrics_path = os.path.join(run_dir, 'metrics.json')
        config_path = os.path.join(run_dir, 'config.json')

        if not os.path.isfile(metrics_path):
            print(f"Skipping {run_dir} (missing metrics.json)")
            continue

        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"Skipping {run_dir} (metrics parse error: {e})")
            continue

        # Fix list/array to scalar: take LAST element
        for key in ['final_reward_correlation', 'final_value_correlation', 'final_policy_agreement',
                   'reward_correlation', 'value_correlation', 'policy_agreement']:
            val = metrics.get(key)
            if isinstance(val, (list, tuple)) and len(val) > 0:
                metrics[key] = float(val[-1])

        try:
            with open(config_path) as f:
                nested_cfg = json.load(f)
            cfg_flat = flatten_config(nested_cfg)
        except Exception as e:
            print(f"Skipping {run_dir} (config parse error: {e})")
            continue

        method = get(cfg_flat, 'irl.method', 'unknown')
        scenario = label_scenario(cfg_flat)

        # Extract final values, fallback to last of series
        record = {
            'run_dir': run_dir,
            'method': method,
            'scenario': scenario,
            'final_wall_time_sec': _last(get(metrics, 'final_wall_time_sec') or get(metrics, 'wall_time_sec')),
            'final_env_steps': _last(get(metrics, 'final_env_steps') or get(metrics, 'env_steps')),
            'final_reward_correlation': _last(get(metrics, 'final_reward_correlation') or get(metrics, 'reward_correlation')),
            'final_value_correlation': _last(get(metrics, 'final_value_correlation') or get(metrics, 'value_correlation')),
            'final_policy_agreement': _last(get(metrics, 'final_policy_agreement') or get(metrics, 'policy_agreement')),
        }
        records.append(record)

    return pd.DataFrame.from_records(records)

def plot_tradeoffs(csv_path: Optional[str], roots: Optional[list], metric: str,
                   x_axis: str, facet: str, out_dir: str, facet_rows: bool=False):
    """Create tradeoff scatter plots."""
    setup_thesis_style()

    if csv_path and os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)

        # Derive scenario if missing
        if 'scenario' not in df.columns:
            print("'scenario' column missing in CSV — deriving it.")
            scenarios = []

            if 'run_dir' in df.columns:
                for _, row in df.iterrows():
                    run_dir = row['run_dir']
                    try:
                        with open(os.path.join(run_dir, 'config.json')) as f:
                            cfg_flat = flatten_config(json.load(f))
                        scenarios.append(label_scenario(cfg_flat))
                    except Exception:
                        scenarios.append('unknown')
            else:
                def _derive(row):
                    # Best-effort inference from CSV fields if present
                    if pd.notna(row.get('heldout_region')): return 'heldout'
                    env_name = row.get('env_name') or row.get('env.name') or ''
                    conf = bool(row.get('confounder', False)) or bool(row.get('env_confounded', False))
                    test_z = row.get('test_z', None)
                    slip = row.get('slip_prob') or row.get('env_slip_prob') or 0
                    demos = row.get('expert_num_trajectories') or row.get('expert.num_trajectories') or 20
                    rtype = row.get('reward_type') or row.get('env_reward_type') or ''
                    if env_name == 'ConfoundedGridWorld' or conf:
                        return 'confounded_crossZ' if pd.notna(test_z) else 'confounded'
                    if slip and float(slip) > 0: return 'noisy'
                    try:
                        if int(demos) <= 10: return 'fewshot'
                    except Exception:
                        pass
                    if rtype == 'shaped': return 'shaped'
                    return 'baseline'
                scenarios = [ _derive(row) for _, row in df.iterrows() ]
            df['scenario'] = scenarios

    elif roots:
        run_dirs = find_run_dirs(roots)
        df = load_data_from_metrics(run_dirs)
    else:
        raise RuntimeError("Either --csv or --roots with valid paths must be provided")

    if df.empty:
        print("No valid data to plot.")
        return

    # Ensure facet exists
    if facet not in df.columns:
        print(f"Facet column '{facet}' missing — falling back to 'method'.")
        facet = 'method'

    # Coerce numeric cols; drop non-numeric rows
    for col in [metric, x_axis]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    needed_cols = ['method', facet, metric, x_axis]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        print(f"Missing columns for plot: {missing}")
        return

    df = df.dropna(subset=needed_cols)

    # Ensure numeric values for scatter plot
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    df[x_axis] = pd.to_numeric(df[x_axis], errors='coerce')
    df = df.dropna(subset=[metric, x_axis])

    facets = sorted(df[facet].unique())
    # First log line with effective options
    print(f"[tradeoffs] source={'CSV:'+csv_path if csv_path else 'roots'} facet={facet} metric={metric} x={x_axis} "
          f"facets={len(facets)} facet_rows={facet_rows}")

    # Layout branch
    if facet_rows:
        ncol, nrow = 1, max(1, len(facets))
        figsize = (6, 3.8 * nrow)
    else:
        ncol = min(3, len(facets))
        nrow = (len(facets) + ncol - 1) // ncol
        figsize = (5 * ncol, 4 * nrow)

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    counts_sidecar: Dict[str, Dict[str, int]] = {}

    # Plot each facet
    idx = 0
    for facet_val in facets:
        # Safety: ensure we don't index past the grid if some facets become empty after coercion
        if idx >= nrow * ncol:
            break
        ax = axes.flat[idx]
        subdf = df[df[facet] == facet_val]
        if subdf.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"no data for {facet}={facet_val}", ha="center", va="center", fontsize=9)
            idx += 1
            continue

        # Scatter by method
        method_counts: Dict[str, int] = {}
        for method, gdf in subdf.groupby('method'):
            method_counts[method] = len(gdf)
            ax.scatter(gdf[x_axis], gdf[metric], label=method, alpha=0.7, s=50)

        # Title with per-method Ns
        counts_str = ", ".join([f"{m}: N={n}" for m, n in sorted(method_counts.items())])
        ax.set_title(f"{facet_val}" + (f"  ({counts_str})" if counts_str else ""))

        # Axes labels
        ax.set_xlabel(x_axis.replace('_', ' ').title() +
                     (" (seconds)" if 'time' in x_axis else " (steps)" if 'steps' in x_axis else ""))
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # Legend (deduplicated)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            ax.legend(uniq.values(), uniq.keys(), fontsize=8)

        # Collect counts for sidecar if in row-facet mode
        if facet_rows:
            counts_sidecar[str(facet_val)] = {str(m): int(n) for m, n in method_counts.items()}

        idx += 1

    # Hide extra axes (if any)
    for ax in axes.flat[idx:]:
        ax.set_visible(False)

    out_path = os.path.join(out_dir, f"accuracy_vs_{x_axis}.png")
    os.makedirs(out_dir, exist_ok=True)
    save_figure(fig, out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', nargs='*', help='Root directories with runs')
    parser.add_argument('--csv', default=None, help='Optional CSV file path')
    parser.add_argument('--metric', required=True,
                       choices=['final_reward_correlation', 'final_value_correlation', 'final_policy_agreement'],
                       help='Metric to plot')
    parser.add_argument('--x', required=True,
                       choices=['final_wall_time_sec', 'final_env_steps'],
                       help='X-axis metric')
    parser.add_argument('--facet', default='scenario', help='Facet by this column')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--facet_rows', action='store_true',
                        help='If set, layout becomes one row per facet (usually scenario).')
    args = parser.parse_args()

    plot_tradeoffs(args.csv, args.roots, args.metric, args.x, args.facet, args.out, facet_rows=args.facet_rows)

if __name__ == '__main__':
    main()
