import os
import json
import glob
import argparse
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd

from collections import defaultdict
from visualisation.utils_config import load_config_with_fallback, get, find_run_dirs
from visualisation.style import setup_thesis_style, save_figure, method_label
from visualisation.scenario import label_scenario


# Default experiment roots for discovery (used when --roots is sparse/absent)
DEFAULT_ROOTS = [
    "results/gridworld_baselines",
    "results/confounded",
    "results/generalization",
    "results/airl_hparams",
    "results/causal_airl_hparams",
    "results/airl_scenarios",
    "results/causal_airl_scenarios",
    "results/scaling",
]

def _dedup(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

# Define x-key aliases for robust lookup
X_KEY_ALIASES = {
    'expert.num_trajectories': [
        'expert.n_trajectories', 'num_trajectories', 'config.expert.num_trajectories', 'cfg.expert.num_trajectories',
        'train.n_demos', 'expert.demos', 'expert.n_demos'
    ],
    'expert.n_trajectories': ['expert.num_trajectories', 'num_trajectories', 'config.expert.num_trajectories', 'cfg.expert.num_trajectories', 'train.n_demos', 'expert.demos', 'expert.n_demos'],
    'num_trajectories': ['expert.num_trajectories', 'expert.n_trajectories', 'config.expert.num_trajectories', 'cfg.expert.num_trajectories', 'train.n_demos', 'expert.demos', 'expert.n_demos'],
    'config.expert.num_trajectories': ['expert.num_trajectories', 'expert.n_trajectories', 'num_trajectories', 'train.n_demos', 'expert.demos', 'expert.n_demos'],
    'train.n_demos': ['expert.num_trajectories', 'expert.n_trajectories', 'num_trajectories'],
    'expert.demos': ['expert.num_trajectories', 'expert.n_trajectories', 'num_trajectories'],
    'expert.n_demos': ['expert.num_trajectories', 'expert.n_trajectories', 'num_trajectories'],
}

METRIC_ALIASES = {
    'final_reward_correlation': ['reward_correlation_final', 'final_reward_corr', 'reward_corr_final'],
    # Spearman (rank fidelity) alias for CSVs that log without 'final_'
    'final_reward_spearman': ['reward_spearman'],
    # Add weighted fallbacks for policy agreement
    'final_policy_agreement': [
        'policy_agreement_final',
        'final_policy_agreement_mean',
        'final_policy_agreement_weighted',
        'policy_agreement_weighted',
        # base (non-final) alias for older CSVs
        'policy_agreement',
    ],
    # Add weighted fallbacks for value correlation
    'final_value_correlation': [
        'final_value_correlation_weighted', 'value_correlation_weighted',
        # base (non-final) alias for older CSVs
        'value_correlation',
    ],
}

def _sanitize(s):
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s))[:200]

def _csv_looks_summarised(df: pd.DataFrame) -> bool:
    cols_lower = [str(c).lower() for c in df.columns]
    cols = set(cols_lower)
    # Heuristics: presence of generic 'mean/lo/hi' or any *_mean columns
    if {'mean', 'lo', 'hi'}.issubset(cols):
        return True
    if any(c.endswith(('_mean', '_lo', '_hi')) for c in cols_lower):
        return True
    # Additional heuristics for pivot-style summary CSVs (experiment_summary.csv):
    # - No per-run identifier
    # - Presence of unnamed columns from a pivot
    # - Has 'scenario'/'method' but lacks raw run_dir
    if 'run_dir' not in df.columns:
        if any(str(c).startswith('Unnamed') for c in df.columns) or any(k in cols for k in ['scenario', 'method']):
            return True
    return False

def _load_config_flat_or_keys(run_dir: str, needed_keys):
    """
    Load a config for run_dir and return a dict with at least the needed dotted keys.
    Tries config_flat.json first; falls back to load_config_with_fallback + get().
    """
    result = {}
    flat_path = os.path.join(run_dir, 'config_flat.json')
    try:
        cfg = None
        if os.path.exists(flat_path):
            with open(flat_path, 'r') as f:
                flat = json.load(f)
            for k in needed_keys:
                if k in flat:
                    result[k] = flat[k]
        else:
            cfg = load_config_with_fallback(run_dir)
            for k in needed_keys:
                result[k] = get(cfg, k)

        # Always include method if discoverable
        if 'irl.method' not in result:
            try:
                cfg = cfg if 'cfg' in locals() else load_config_with_fallback(run_dir)
                result['irl.method'] = get(cfg, 'irl.method') or get(cfg, 'method')
            except Exception:
                pass

        # Always include a synthesized scenario label if requested
        if 'scenario' in needed_keys:
            try:
                cfg = cfg if cfg is not None else load_config_with_fallback(run_dir)
                result['scenario'] = label_scenario(cfg)
            except Exception:
                pass

    except Exception as e:
        print(f"Warning: failed reading config for {run_dir}: {e}")
    return result

def enrich_df_with_configs(df: pd.DataFrame, x_key: str, groupby_keys=None, roots=None):
    """
    Enrich df rows using config keys pulled from run directories referenced by df['run_dir']
    and/or discovered via roots. Also attempts to populate missing x_key via aliases.
    """
    if df is None or df.empty:
        return df

    needed = set(['irl.method', 'env.name', 'env.confounder_value', 'expert.num_trajectories', 'scenario'])
    if groupby_keys:
        needed.update(groupby_keys)

    # Build set of candidate run dirs from df and roots
    run_dirs = set()
    if 'run_dir' in df.columns:
        run_dirs.update(df['run_dir'].dropna().astype(str).tolist())
    if roots:
        discovered = find_run_dirs(roots)
        run_dirs.update(discovered)
        print(f"[discover] Found {len(discovered)} runs from roots={roots}")

    if not run_dirs:
        return df

    # Cache configs keyed by both absolute path and basename
    cache = {}
    for rd in run_dirs:
        vals = _load_config_flat_or_keys(rd, needed)
        cache[rd] = vals
        cache[os.path.basename(rd)] = vals

    # Ensure all needed columns exist in df
    for k in needed:
        if k not in df.columns:
            df[k] = None

    # Row-wise fill from cache using run_dir or its basename
    def _fill(row):
        rd = str(row.get('run_dir', '') or '')
        vals = None
        if rd:
            vals = cache.get(rd) or cache.get(os.path.basename(rd))

        # Try alternative identifiers if run_dir was not embedded in CSV
        if vals is None:
            for id_key in ('run_id', 'experiment_id', 'id'):
                rid = row.get(id_key)
                if isinstance(rid, str) and rid:
                    vals = cache.get(rid) or cache.get(os.path.basename(rid))
                    if vals:
                        break
        if vals:
            for k, v in vals.items():
                if row.get(k) is None or (isinstance(row.get(k), float) and pd.isna(row.get(k))):
                    row[k] = v

            # Normalise method column
            if 'method' not in row or pd.isna(row.get('method')):
                m = vals.get('irl.method')
                if m is not None:
                    row['method'] = m

            # Backfill scenario if missing
            if row.get('scenario') in (None, '', float('nan')):
                s = vals.get('scenario')
                if s is not None:
                    row['scenario'] = s

        return row
    df = df.apply(_fill, axis=1)

    # Populate x_key from aliases if still missing or all-NaN
    if x_key not in df.columns or df[x_key].isna().all():
        for alias in X_KEY_ALIASES.get(x_key, []):
            if alias in df.columns and not df[alias].isna().all():
                df[x_key] = df[alias]
                break
    return df

def load_from_csv(csv_paths, x_key, method_filter=None, groupby_keys=None):
    """Load data directly from CSV files, bypassing per-run JSON parsing."""
    dfs = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"Warning: CSV not found: {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # If CSV appears summarised, auto-fallback to detailed_results.csv
    if _csv_looks_summarised(df):
        fallback_tried = False
        for csv_path in csv_paths:
            cand = os.path.join(os.path.dirname(csv_path), 'detailed_results.csv')
            if os.path.exists(cand):
                try:
                    df = pd.read_csv(cand)
                    print(f"Detected summarised CSV; auto-falling back to {cand}")
                    fallback_tried = True
                    break
                except Exception:
                    pass
        if not fallback_tried:
            default_cand = os.path.join('results', 'validation', 'detailed_results.csv')
            if os.path.exists(default_cand):
                try:
                    df = pd.read_csv(default_cand)
                    print(f"Detected summarised CSV; auto-falling back to {default_cand}")
                    fallback_tried = True
                except Exception:
                    pass
        if not fallback_tried:
            print("CSV looks summarised (mean/lo/hi present) and no detailed_results.csv found nearby — "
                  "supply detailed_results.csv or use --roots.")
            return pd.DataFrame()

    else:
        # Even if not flagged summarised, a CSV without run_dir usually can't be enriched.
        # Gracefully try a nearby detailed_results.csv if present.
        if 'run_dir' not in df.columns:
            for csv_path in csv_paths:
                cand = os.path.join(os.path.dirname(csv_path), 'detailed_results.csv')
                if os.path.exists(cand):
                    try:
                        df = pd.read_csv(cand)
                        print(f"No 'run_dir' in CSV; falling back to {cand}")
                        break
                    except Exception:
                        pass

    # Filter by method if requested
    if method_filter:
        meth_col = None
        for c in ['method', 'irl.method', 'config.irl.method']:
            if c in df.columns:
                meth_col = c
                break
        if meth_col is None and 'run_dir' in df.columns:
            # derive from run_dir heuristically
            df['method'] = df['run_dir'].astype(str).str.contains('causal', case=False).map({True: 'Causal-AIRL', False: 'AIRL'})
            meth_col = 'method'
        if meth_col:
            df = df[df[meth_col].astype(str).str.lower() == method_filter.lower()]

    # Check x_key exists, try aliases if not found
    if x_key not in df.columns:
        # Try aliases for x_key (do not early-return; enrichment step may populate it)
        for alias in X_KEY_ALIASES.get(x_key, []):
            if alias in df.columns:
                df[x_key] = df[alias]
                break
        # Any column ending with .num_trajectories
        if x_key not in df.columns:
            tail_cols = [c for c in df.columns if str(c).endswith('.num_trajectories')]
            if tail_cols:
                df[x_key] = df[tail_cols[0]]
            else:
                print(f"Note: x_key '{x_key}' not found in CSV; will attempt to enrich from configs.")

    # Ensure a 'method' column exists (for grouping/legends)
    if 'method' not in df.columns:
        for c in ['irl.method', 'config.irl.method']:
            if c in df.columns:
                df['method'] = df[c]
                break
        if 'method' not in df.columns:
            if 'run_dir' in df.columns:
                df['method'] = df['run_dir'].astype(str).str.contains('causal', case=False).map({True: 'Causal-AIRL', False: 'AIRL'})
            else:
                df['method'] = 'unknown'
                print("Note: no 'method' column found in CSV; defaulting to 'unknown'.")

    # Ensure groupby columns are present or synthesized
    if groupby_keys:
        for group_key in groupby_keys:
            if group_key not in df.columns:
                # Last-resort synthesis for 'scenario' if requested; otherwise leave missing
                if group_key == 'scenario':
                    # Compose a lightweight scenario label from available columns
                    parts = []
                    for k in ['env.name', 'heldout.region', 'confounded', 'expert.num_trajectories', 'config.expert.num_trajectories']:
                        if k in df.columns:
                            parts.append(df[k].astype(str))
                    if parts:
                        df['scenario'] = pd.Series(["|".join(vals) for vals in zip(*parts)], index=df.index)

    return df

def load_ablation_data(roots, x_key, method_filter=None, groupby_keys=None):
    records = []

    run_dirs = find_run_dirs(roots)

    for run_dir in run_dirs:
        config_path = os.path.join(run_dir, 'config.json')
        metrics_path = os.path.join(run_dir, 'metrics.json')
        if not os.path.exists(config_path) or not os.path.exists(metrics_path):
            print(f"Skipping {run_dir} (missing config or metrics.json)")
            continue

        try:
            cfg = load_config_with_fallback(run_dir)
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"Skipping {run_dir} due to error: {e}")
            continue

        # Filter by method if requested
        method = get(cfg, 'irl.method') or get(cfg, 'method') or 'unknown'
        if method_filter and method.lower() != method_filter.lower():
            continue

        record = {'method': method, 'run_dir': run_dir}

        # Add groupby keys if specified
        if groupby_keys:
            for k in groupby_keys:
                record[k] = get(cfg, k)

        # Add scenario
        record['scenario'] = label_scenario(cfg)

        # Get x-axis value from config or metrics
        x_val = get(cfg, x_key)
        if x_val is None:
            # Try aliases from config first
            for alias in X_KEY_ALIASES.get(x_key, []):
                x_val = get(cfg, alias)
                if x_val is not None:
                    break

            # Fallback to metrics.json
            if x_val is None:
                x_val = metrics.get(x_key)

        if x_val is None:
            print(f"Skipping {run_dir}: {x_key} not found")
            continue

        record[x_key] = x_val

        # Add all metrics, handling lists by taking last value
        for k, v in metrics.items():
            if isinstance(v, list) and len(v) > 0:
                v = v[-1]  # Use last value for time series
            record[k] = v

        records.append(record)

    return pd.DataFrame(records) if records else pd.DataFrame()

def create_placeholder(save_path, message):
    """Create a placeholder image with centered message when no data is available."""
    setup_thesis_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14,
            transform=ax.transAxes, wrap=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    save_figure(fig, save_path)
    plt.close(fig)

def plot_metric(df, x_key, y_key, save_path, groupby_keys=None):
    setup_thesis_style()

    if y_key not in df.columns:
        print(f"Metric '{y_key}' not found in data; skipping figure.")
        return

    if df.empty:
        create_placeholder(save_path, f"No data for metric={y_key}, x={x_key}")
        return

    plt.figure(figsize=(8, 6))

    # Create grouping columns based on method and optional groupby keys
    if groupby_keys:
        # Create composite grouping key
        def make_group_label(row):
            method = row.get('method', 'unknown')
            parts = [f"{key}={row.get(key, 'N/A')}" for key in groupby_keys]
            return f"{method} | {', '.join(parts)}"

        df['_group_label'] = df.apply(make_group_label, axis=1)
        group_col = '_group_label'
    else:
        group_col = 'method'

    # Check if we have any valid data points
    valid_data = df.dropna(subset=[x_key, y_key])
    if valid_data.empty:
        groupby_str = f", groupby={','.join(groupby_keys)}" if groupby_keys else ""
        create_placeholder(save_path, f"No data for metric={y_key}, x={x_key}{groupby_str}")
        return

    for group_val, group_df in valid_data.groupby(group_col):
        # Drop NaNs for plotting
        data = group_df.dropna(subset=[x_key, y_key])
        if len(data) == 0:
            continue
        label = group_val
        if not groupby_keys and group_col == 'method':
            label = method_label(group_val)
        plt.plot(data[x_key], data[y_key], marker='o', label=label, alpha=0.8)

    plt.xlabel(x_key.replace('_', ' ').title())
    plt.ylabel(y_key.replace('_', ' ').title())
    title = f'{y_key.replace("_", " ").title()} vs {x_key.replace("_", " ").title()}'
    if groupby_keys:
        title += f' — grouped by {", ".join(groupby_keys)}'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure(plt.gcf(), save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot ablation summaries")
    parser.add_argument('--roots', nargs='+', required=False, help='Root directories with runs')
    parser.add_argument('--x', required=True, help="Config dotted key for x-axis (e.g. expert.num_trajectories)")
    parser.add_argument('--metrics', nargs='+', required=True, help="Metrics to plot from metrics.json")
    parser.add_argument('--groupby', nargs='*', default=None, help="List of config dotted keys to group by (comma or space separated)")
    parser.add_argument('--method_filter', default=None, help="Filter by method name")
    parser.add_argument('--csv', nargs='+', default=None, help="CSV files to load data from (bypasses --roots)")
    parser.add_argument('--out', required=True, help="Output directory under results/figures/summary")

    args = parser.parse_args()

    # Normalize groupby: handle both comma-separated and space-separated
    groupby_keys = []
    if args.groupby:
        if len(args.groupby) == 1 and ',' in args.groupby[0]:
            # Single comma-separated string
            groupby_keys = [k.strip() for k in args.groupby[0].split(',') if k.strip()]
        else:
            # Space-separated list
            groupby_keys = args.groupby

    # Create output directory
    os.makedirs(args.out, exist_ok=True)

    # Effective roots: user-supplied + defaults (dedup)
    eff_roots = []
    if args.roots:
        eff_roots.extend(args.roots)
    eff_roots.extend(DEFAULT_ROOTS)
    eff_roots = _dedup([r for r in eff_roots if r])
    if not args.roots:
        # provide discoverable roots even for CSV-only workflows
        args.roots = eff_roots
    print(f"[inputs] csv={args.csv or []} | roots={eff_roots}")

    # Load data:
    # If CSV is provided, load it; if --roots also provided, we will enrich with configs from runs.
    df = pd.DataFrame()
    if args.csv:
        df = load_from_csv(args.csv, args.x, args.method_filter, groupby_keys)
    if (df is None or df.empty) and args.roots:
        # Fall back to per-run loading
        df = load_ablation_data(args.roots, args.x, args.method_filter, groupby_keys)

    if df is None:
        df = pd.DataFrame()

    # Enrich with configs if possible (handles both CSV-only and CSV+roots)
    if not df.empty:
        before_cols = set(df.columns)
        df = enrich_df_with_configs(df, args.x, groupby_keys=groupby_keys, roots=eff_roots)
        added_cols = sorted(list(set(df.columns) - before_cols))
        print(f"[enrich] Added columns from configs: {', '.join(added_cols) if added_cols else '(none)'}")

    rows_in_before_numeric = len(df)

    # If neither source provided, error out
    if df.empty and not (args.csv or args.roots):
        parser.error("Must provide either --roots or --csv")

    # Apply metric aliases and coerce numeric types
    def _apply_metric_aliases(df, metrics):
        dropped_total = 0
        for m in metrics:
            if m not in df.columns:
                for alt in METRIC_ALIASES.get(m, []):
                    if alt in df.columns:
                        df[m] = df[alt]
                        break

        # x aliases handled later; ensure a column exists so to_numeric works
        # Ensure x column exists before coercion to avoid KeyError
        if args.x not in df.columns:
            # Try one more time via aliases already loaded into df
            for alias in X_KEY_ALIASES.get(args.x, []):
                if alias in df.columns:
                    df[args.x] = df[alias]
                    break
        if args.x not in df.columns:
            df[args.x] = pd.Series([float('nan')] * len(df))

        # Coerce x and metrics to numeric (non-numeric → NaN)
        df[args.x] = pd.to_numeric(df[args.x], errors='coerce')

        # Report diagnostics **before** dropping
        n = max(len(df), 1)
        nx = int(df[args.x].notna().sum())
        pct_x = 100.0 * nx / n
        print(f"[diag] rows={n} | numeric x={nx} ({pct_x:.1f}%)")
        metric_cols = [m for m in args.metrics if m in df.columns]
        for m in metric_cols:
            nn = int(pd.to_numeric(df[m], errors='coerce').notna().sum())
            print(f"[diag] metric '{m}': numeric rows={nn} ({100.0*nn/n:.1f}%)")

        before = len(df)

        # Drop rows only if x is NaN OR all requested metrics are NaN (relaxed rule)

        if metric_cols:
            mask_all_nan = (
                df[metric_cols].apply(pd.to_numeric, errors='coerce').isna().all(axis=1)
            ) | df[args.x].isna()
            dropped_total = int(mask_all_nan.sum())
            df.drop(df.index[mask_all_nan], inplace=True)
        if dropped_total:
            print(f"[drop] Dropped {dropped_total} row(s) where x was NaN or all requested metrics were NaN.")
        kept = len(df)
        print(f"[summary] kept_rows={kept} (from {rows_in_before_numeric}) after relaxed drop rule")
        return df

    if not df.empty:
        df = _apply_metric_aliases(df, args.metrics)

    # Diagnostics for missing keys post-join
    missing_reasons = []
    if df.empty:
        missing_reasons.append("no rows after loading/enrichment")
    else:
        if args.x not in df.columns:
            missing_reasons.append(f"x_key '{args.x}' missing after CSV+config join")
        elif df[args.x].isna().all():
            missing_reasons.append(f"x_key '{args.x}' present but all values NaN")
        if groupby_keys:
            missing_gb = [g for g in groupby_keys if g not in df.columns]
            if missing_gb:
                missing_reasons.append("missing groupby keys: " + ", ".join(missing_gb))

    if df.empty or missing_reasons:
        reason = "; ".join(missing_reasons) if missing_reasons else "no data after normalization"
        print(f"No data found for plotting after CSV/roots loading and normalization — {reason}. "
              f"Searched roots={eff_roots}")
        # Only now (truly empty) emit placeholders for each requested metric
        for metric in args.metrics:
            suffix = ""
            if groupby_keys:
                suffix = f"__by_{_sanitize('-'.join(groupby_keys))}"
            save_path = os.path.join(args.out, f"{_sanitize(metric)}_vs_{_sanitize(args.x)}{suffix}.png")
            groupby_str = f", groupby={','.join(groupby_keys)}" if groupby_keys else ""
            create_placeholder(save_path, f"No data for metric={metric}, x={args.x}{groupby_str}\nReason: {reason}")
        return

    # Plot each requested metric if it has any usable data; otherwise skip quietly
    for metric in args.metrics:
        if metric not in df.columns or df.dropna(subset=[args.x, metric]).empty:
            print(f"Skipping metric '{metric}' — no usable rows after coercion/aliasing.")
            continue
        suffix = ""
        if groupby_keys:
            suffix = f"__by_{_sanitize('-'.join(groupby_keys))}"
        save_path = os.path.join(args.out, f"{_sanitize(metric)}_vs_{_sanitize(args.x)}{suffix}.png")
        plot_metric(df, args.x, metric, save_path, groupby_keys)

    print(f"[summary] Plotted {len([m for m in args.metrics if m in df.columns])} metric(s) "
          f"over {len(df)} row(s). Output → {args.out}")

if __name__ == '__main__':
    main()
