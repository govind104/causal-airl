import argparse
import os
import json
import csv
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from visualisation.utils_config import find_run_dirs, flatten_config, get
from visualisation.style import setup_thesis_style, save_figure
from visualisation.scenario import label_scenario


CI_LEVEL = 0.95

def plot_crossz_bars(roots, out_dir):
    """Plot cross-Z generalization bars."""
    setup_thesis_style()
    run_dirs = find_run_dirs(roots)

    # Collect data by scenario and method
    data = {}

    print(f"[crossz] roots={roots} | discovered_runs={len(run_dirs)}")

    for run_dir in run_dirs:
        metrics_path = os.path.join(run_dir, 'metrics.json')
        config_path = os.path.join(run_dir, 'config.json')

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

        method = get(cfg_flat, 'irl.method', 'unknown')
        scenario = label_scenario(cfg_flat)

        # Look for cross-Z metrics
        cross_z_metrics = {}
        for key in metrics.keys():
            if key.startswith('cross_z_from_'):
                cross_z_metrics[key] = metrics[key]

        if not cross_z_metrics:
            continue

        # Normalize scenario label when cross-Z metrics are present
        scenario_key = f"{scenario}"
        if cross_z_metrics:
            base = scenario.split('_')[0] if '_' in scenario else scenario
            scenario_key = f"{base}_crossz"

        if scenario_key not in data:
            data[scenario_key] = {}
        if method not in data[scenario_key]:
            data[scenario_key][method] = {}

        for metric_key, value in cross_z_metrics.items():
            # Ensure numeric scalar; take LAST if list
            if isinstance(value, list) and len(value) > 0:
                try:
                    value = float(value[-1])
                except Exception:
                    value = None
            else:
                try:
                    value = float(value)
                except Exception:
                    value = None

            if value is None:
                continue
            value = max(0.0, min(1.0, value))
            # Aggregate as lists across runs
            if metric_key not in data[scenario_key][method]:
                data[scenario_key][method][metric_key] = []
            data[scenario_key][method][metric_key].append(value)

    if not data:
        print("No cross-Z data found to plot.")
        return

    # Plot each scenario
    for scenario, scenario_data in data.items():
        # Collect directions and diagnostics counts
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = list(scenario_data.keys())
        n_methods = len(methods)

        # Get all unique cross-Z directions
        all_directions = set()
        for method_data in scenario_data.values():
            all_directions.update(method_data.keys())
        directions = sorted(all_directions)

        if not directions:
            continue

        x = np.arange(len(directions))
        width = 0.8 / n_methods

        # Stats and logging dict for Ns
        counts_log = {d: {} for d in directions}
        rows_for_csv = []  # optional sidecar

        for i, method in enumerate(methods):
            values = []
            means = []
            lower_err = []
            upper_err = []
            for direction in directions:
                vals = scenario_data[method].get(direction, [])
                vals = [v for v in vals if v is not None]
                n = len(vals)
                counts_log[direction][method] = n
                if n == 0:
                    mu, lo, hi = 0.0, 0.0, 0.0
                elif n == 1:
                    mu = float(vals[0])
                    lo, hi = mu, mu
                else:
                    arr = np.asarray(vals, dtype=float)
                    mu = float(np.mean(arr))
                    sd = float(np.std(arr, ddof=1))
                    se = sd / np.sqrt(n)
                    lo = mu - 1.96 * se
                    hi = mu + 1.96 * se
                # clip CIs into [0,1]
                lo = max(0.0, min(1.0, lo))
                hi = max(0.0, min(1.0, hi))
                means.append(mu)
                lower_err.append(max(0.0, mu - lo) if n >= 2 else 0.0)
                upper_err.append(max(0.0, hi - mu) if n >= 2 else 0.0)
                # sidecar: hard-clip mean/lo/hi; include CI provenance
                rows_for_csv.append({
                    "direction": direction,
                    "method": method,
                    "mean": max(0.0, min(1.0, mu)),
                    "lo":   max(0.0, min(1.0, lo)),
                    "hi":   max(0.0, min(1.0, hi)),
                    "n": n,
                    "ci_level": CI_LEVEL,
                })

            offset = (i - n_methods/2 + 0.5) * width
            yerr = np.vstack([lower_err, upper_err]) if any(e > 0 for e in lower_err+upper_err) else None
            bars = ax.bar(x + offset, means, width, label=method, alpha=0.8,
                          yerr=yerr, error_kw=dict(capsize=3, lw=1))

            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                # Annotate N under/above as small text
                # Pick a position slightly inside the bar if tall, else above
                n_here = scenario_data[method].get(directions[j], [])
                n_here = len(n_here) if n_here is not None else 0
                yN = height - 0.05 if height > 0.2 else height + 0.06
                ax.text(bar.get_x() + bar.get_width()/2., yN,
                        f'N={n_here}', ha='center', va='center', fontsize=7, color='black')

        # Format direction labels
        clean_labels = []
        for direction in directions:
            # Convert cross_z_from_1_to_0 to "1→0"
            parts = direction.replace('cross_z_from_', '').split('_to_')
            if len(parts) == 2:
                clean_labels.append(f"{parts[0]}→{parts[1]}")
            else:
                clean_labels.append(direction)

        ax.set_xlabel('Cross-Z Direction')
        ax.set_ylabel('Policy Agreement')
        ax.set_title(f'Cross-Z Generalization: {scenario}')
        ax.set_xticks(x)
        ax.set_xticklabels(clean_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Sanitize file name, force single canonical “…_crossz_bars”
        safe_scenario = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in scenario.lower())
        fname = f'{safe_scenario}_bars.png'
        out_path = os.path.join(out_dir, fname)
        os.makedirs(out_dir, exist_ok=True)
        save_figure(fig, out_path)
        print(f"[crossz] {scenario}: counts={counts_log} ci={CI_LEVEL} -> {out_path}")

        # Sidecar CSV with stats (+ CI provenance)
        try:
            sidecar_csv = os.path.join(out_dir, f"{safe_scenario}_bars_stats.csv")
            with open(sidecar_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["direction","method","mean","lo","hi","n","ci_level"])
                w.writeheader()
                for row in rows_for_csv:
                    w.writerow(row)
            print(f"[crossz] wrote sidecar: {sidecar_csv}")
        except Exception as e:
            print(f"[warn] failed to write sidecar CSV: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', nargs='+', required=True, help='Root directories with runs')
    parser.add_argument('--out', required=True, help='Output directory')
    args = parser.parse_args()

    plot_crossz_bars(args.roots, args.out)

if __name__ == '__main__':
    main()
