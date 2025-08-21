import argparse
import os
import json
import csv
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualisation.utils_config import find_run_dirs, flatten_config, get
from visualisation.style import setup_thesis_style, save_figure, method_label, set_method_color_cycle
from visualisation.scenario import label_scenario


CI_LEVEL = 0.95

def plot_crossz_bars(roots, out_dir, perz_csv=None):
    """Plot cross-Z generalization bars."""
    setup_thesis_style()
    run_dirs = find_run_dirs(roots)

    # Collect data by scenario and method
    data = {}

    # Optional per-Z CSV for worst-Z overlays
    perz_df = None
    if perz_csv and os.path.exists(perz_csv):
        try:
            perz_df = pd.read_csv(perz_csv)
            # Ensure numeric for filtering/aggregates
            for c in ["env.slip_prob", "final_reward_spearman", "final_value_correlation"]:
                if c in perz_df.columns:
                    perz_df[c] = pd.to_numeric(perz_df[c], errors="coerce")
        except Exception as e:
            print(f"[crossz] warn: failed to read perZ CSV '{perz_csv}': {e}")
            perz_df = None

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
            # Pull slip prob for scenario disambiguation
            slip = float(get(cfg_flat, 'env.slip_prob', 0.0) or 0.0)
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
            scenario_key = f"{base}_crossz_slip{slip:.2f}"

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

        # Stabilize colors across figures by method order
        set_method_color_cycle(methods)

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
        # Optional overlay: worst-Z badges from perZ CSV filtered by scenario slip
        title = f'Cross-Z Generalization: {scenario}'
        if perz_df is not None:
            # Parse slip from scenario key suffix: "..._crossz_slip{xx.xx}"
            slip_match = None
            if "slip" in scenario:
                try:
                    slip_match = float(scenario.rsplit("slip", 1)[-1])
                except Exception:
                    slip_match = None
            if slip_match is not None and "env.slip_prob" in perz_df.columns:
                mask = perz_df["env.slip_prob"].round(2) == round(slip_match, 2)
                sub = perz_df.loc[mask].copy()
                # Compute worst-Z per run (min over z), then mean across runs
                if not sub.empty and "run_path" in sub.columns:
                    grp = sub.groupby("run_path", dropna=False)
                    sp_worst = grp["final_reward_spearman"].min().mean() if "final_reward_spearman" in sub.columns else np.nan
                    # ValueCorr with weighted fallbacks (robust to older perZ CSVs)
                    vc_candidates = ["final_value_correlation",
                                     "final_value_correlation_weighted",
                                    "value_correlation_weighted"]
                    vc_col = next((c for c in vc_candidates if c in sub.columns), None)
                    vc_worst = grp[vc_col].min().mean() if vc_col else np.nan
                else:
                    sp_worst = sub["final_reward_spearman"].min() if "final_reward_spearman" in sub.columns else np.nan
                    vc_candidates = ["final_value_correlation",
                                     "final_value_correlation_weighted",
                                     "value_correlation_weighted"]
                    vc_col = next((c for c in vc_candidates if c in sub.columns), None)
                    vc_worst = sub[vc_col].min() if vc_col else np.nan
                parts = []
                if pd.notna(sp_worst): parts.append(f"Worst-Z Spearman={sp_worst:.3f}")
                if pd.notna(vc_worst): parts.append(f"Worst-Z ValueCorr={vc_worst:.3f}")
                if parts:
                    title = f"{title} — " + " | ".join(parts)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(clean_labels)
        
        # Friendly legend labels
        handles, labels = ax.get_legend_handles_labels()
        labels = [method_label(lab) for lab in labels]
        ax.legend(handles, labels)
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
    parser.add_argument('--perz_csv', default=None, help='Optional per-Z CSV for worst-Z overlays')
    args = parser.parse_args()

    plot_crossz_bars(args.roots, args.out, perz_csv=args.perz_csv)

if __name__ == '__main__':
    main()
