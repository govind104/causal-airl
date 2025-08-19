import os
import argparse
import pandas as pd
import numpy as np

from pathlib import Path
from typing import List
from visualisation.scenario import label_scenario
from visualisation.utils_config import get
from experiments.stats import bootstrap_mean_ci


def flatten_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert list-valued columns to scalars by taking their mean."""
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            print(f"[Info] Flattening list-valued column: {col}")
            df[col] = df[col].apply(lambda x: np.mean(x) if isinstance(x, list) else x)
    return df

def compute_summary_with_ci(df: pd.DataFrame, groupby: list, metrics: list, ci: int = 95):
    """Compute mean ± CI for each group × metric combination"""
    results = []
    rng = np.random.default_rng(42)

    for group_vals, group_df in df.groupby(groupby):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)

        record = {}
        for i, key in enumerate(groupby):
            record[key] = group_vals[i]

        for metric in metrics:
            if metric not in group_df.columns:
                continue

            values = group_df[metric].dropna()
            if len(values) == 0:
                continue

            mean, lo, hi = bootstrap_mean_ci(values.values, ci=ci, B=1000, rng=rng)

            record[f'{metric}_mean'] = mean
            record[f'{metric}_ci_low'] = lo
            record[f'{metric}_ci_high'] = hi
            record[f'{metric}_mean_ci'] = f"{mean:.3f}±[{lo:.3f},{hi:.3f}]"
            record[f'{metric}_n'] = len(values)

        results.append(record)

    return pd.DataFrame(results)


def save_table(df: pd.DataFrame, save_path: str, fmt="csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if fmt == "csv":
        df.to_csv(save_path, index=False)
    else:
        raise ValueError("Unsupported format.")

def _csv_looks_summarised(df: pd.DataFrame) -> bool:
    cols = {str(c).lower() for c in df.columns}
    if {"mean", "lo", "hi"}.issubset(cols):
        return True
    return any(c.endswith(("_mean", "_lo", "_hi")) for c in cols)

def _try_load_csv(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"Loaded {len(df)} records from {path}")
            return df
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
    return None

def load_experiment_csv(csv_paths):
    """Load experiment data from CSV files. If a CSV appears summarised, auto-fallback to detailed_results.csv."""
    dfs = []
    summarised_detected = False
    for csv_path in csv_paths:
        df = _try_load_csv(csv_path)
        if df is None:
            continue
        dfs.append(df)
        # Detect summarised schema
        if _csv_looks_summarised(df):
            summarised_detected = True

    if not dfs:
        raise RuntimeError("No valid CSV files found")

    combined_df = pd.concat(dfs, ignore_index=True)

    if summarised_detected:
        # Prefer sibling detailed_results.csv next to the first provided CSV, else a common default
        fallbacks = []
        for csv_path in csv_paths:
            fallbacks.append(os.path.join(os.path.dirname(csv_path), "detailed_results.csv"))
        fallbacks.append(os.path.join("results", "validation", "detailed_results.csv"))
        for cand in fallbacks:
            ddf = _try_load_csv(cand)
            if ddf is not None:
                print(f"Detected summarised CSV; auto-falling back to raw rows in {cand}")
                return ddf
        raise RuntimeError(
            "Input CSV looks summarised (mean/lo/hi present), but no detailed_results.csv found. "
            "Please supply results/validation/detailed_results.csv or use a raw per-run CSV."
        )

    return combined_df

def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, int]:
    """Coerce selected columns to numeric; drop rows where all selected metrics are NaN or non-numeric."""
    df = df.copy()
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # drop rows where all selected metrics are NaN
    if columns:
        mask_all_nan = df[columns].isna().all(axis=1)
        dropped = int(mask_all_nan.sum())
        df = df.loc[~mask_all_nan].reset_index(drop=True)
    else:
        dropped = 0
    return df, dropped

def _default_tables_dir(p: str) -> str:
    """Ensure outputs live under results/tables/ when user gives a bare filename."""
    if os.path.dirname(p) == '':
        return os.path.join("results", "tables", p)
    return p

def _to_markdown_table(df: pd.DataFrame) -> str:
    """Lightweight Markdown writer (no external deps)."""
    cols: List[str] = list(map(str, df.columns))
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and np.isfinite(v):
                vals.append(f"{v:.6g}")  # clear, non-truncated but compact
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"

def _write_markdown(df: pd.DataFrame, md_path: str):
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_to_markdown_table(df))

def main():
    parser = argparse.ArgumentParser(description="Generate summary tables with confidence intervals")
    parser.add_argument("--csv", nargs='+', required=True, help="Path(s) to summary CSV(s)")
    parser.add_argument("--groupby", nargs='+', required=True, help="Group by keys (comma- or space-separated)")
    parser.add_argument("--metrics", nargs='+', required=True, help="Metric keys to summarise (comma- or space-separated)")
    parser.add_argument("--out", type=str, default="experiment_summary_ci.csv", help="Output CSV path (default under results/tables/)")
    parser.add_argument("--md", type=str, default=None, help="Optional Markdown output path (default mirrors --out with .md)")
    parser.add_argument("--ci", type=int, default=95, help="Confidence interval percentage")

    args = parser.parse_args()

    # Parse groupby: handle both comma-separated and space-separated
    groupby = []
    if len(args.groupby) == 1 and ',' in args.groupby[0]:
        # Single comma-separated string like "scenario,method"
        groupby = [k.strip() for k in args.groupby[0].split(',') if k.strip()]
    else:
        # Space-separated list like "scenario method"
        groupby = args.groupby

    # Parse metrics: handle both comma-separated and space-separated
    metrics = []
    if len(args.metrics) == 1 and ',' in args.metrics[0]:
        # Single comma-separated string
        metrics = [m.strip() for m in args.metrics[0].split(',') if m.strip()]
    else:
        # Space-separated list
        metrics = args.metrics

    # Load data
    df = load_experiment_csv(args.csv)
    df = flatten_list_columns(df)

    # Ensure output path defaults to results/tables/ if no directory given
    out_path = _default_tables_dir(args.out)
    # Decide Markdown companion path
    md_path = _default_tables_dir(args.md) if args.md else _default_tables_dir(
        os.path.splitext(out_path)[0] + ".md"
    )

    # Synthesize scenario whenever missing (requested or not) and include by default as a column
    if 'scenario' not in df.columns:
        print("Adding scenario labels (synthesised)...")
        def _infer_scenario(row):
            # Enhanced scenario synthesis based on multiple column patterns
            heldout_region = row.get('heldout_region') or row.get('eval.heldout_region')
            confounded = bool(row.get('confounded', False)) or bool(row.get('env.confounded', False))
            env_name = str(row.get('env_name', '') or '')
            test_z = row.get('eval.test_z') or row.get('test_z')
            slip_prob = float(row.get('slip_prob', 0.0) or row.get('env.slip_prob', 0.0) or 0.0)
            num_trajectories = row.get('expert.num_trajectories', row.get('num_trajectories', 1000))
            reward_type = str(row.get('env.reward_type', row.get('reward_type', '')) or '')

            if heldout_region is not None:
                return 'heldout'
            if confounded or env_name == 'ConfoundedGridWorld':
                if test_z is not None:
                    return 'confounded_crossZ'
                return 'confounded'
            if slip_prob and slip_prob > 0:
                return 'noisy'
            if reward_type == 'shaped':
                return 'shaped'
            try:
                if num_trajectories is not None and int(num_trajectories) <= 10:
                    return 'fewshot'
            except Exception:
                pass
            return 'baseline'

        df['scenario'] = df.apply(_infer_scenario, axis=1)

    # Ensure 'method' exists (derive from other cols if missing)
    if 'method' not in df.columns:
        for c in ['irl.method', 'config.irl.method']:
            if c in df.columns:
                df['method'] = df[c]
                break
        if 'method' not in df.columns and 'run_dir' in df.columns:
            df['method'] = df['run_dir'].astype(str).str.contains('causal', case=False).map({True: 'Causal-AIRL', False: 'AIRL'})
        if 'method' not in df.columns:
            print("Note: 'method' not found; downstream grouping by method will be ignored unless present.")

    # Check for missing columns
    missing_groupby = [g for g in groupby if g not in df.columns]
    missing_metrics = [m for m in metrics if m not in df.columns]

    if missing_groupby:
        print(f"Warning: Missing groupby columns: {missing_groupby}")
        groupby = [g for g in groupby if g in df.columns]

    if missing_metrics:
        print(f"Warning: Missing metric columns: {missing_metrics}")
        metrics = [m for m in metrics if m in df.columns]

    if not groupby or not metrics:
        # Write placeholder CSV for determinism
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame([{
            "note": "No valid groupby keys or metrics found after schema checks.",
            "requested_groupby": " ".join(groupby),
            "requested_metrics": " ".join(metrics),
        }]).to_csv(out_path, index=False)
        # Mirror placeholder to Markdown
        _write_markdown(pd.DataFrame([{
            "note": "No valid groupby keys or metrics found after schema checks.",
            "requested_groupby": " ".join(groupby),
            "requested_metrics": " ".join(metrics),
        }]), md_path)
        print(f"Wrote placeholder CSV to {out_path}")
        print(f"Wrote placeholder Markdown to {md_path}")
        return

    # Coerce metrics to numeric and drop rows where all selected metrics are NaN
    df, dropped = _coerce_numeric_columns(df, metrics)
    if dropped:
        print(f"Dropped {dropped} rows with non-numeric/NaN metrics.")
    if df.empty:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame([{
            "note": "No usable rows after numeric coercion.",
            "requested_groupby": " ".join(groupby),
            "requested_metrics": " ".join(metrics),
        }]).to_csv(out_path, index=False)
        _write_markdown(pd.DataFrame([{
            "note": "No usable rows after numeric coercion.",
            "requested_groupby": " ".join(groupby),
            "requested_metrics": " ".join(metrics),
        }]), md_path)
        print(f"Wrote placeholder CSV to {out_path}")
        print(f"Wrote placeholder Markdown to {md_path}")
        return

    # Generate summary table with CIs
    summary_df = compute_summary_with_ci(df, groupby, metrics, args.ci)

    # Drop groups where n==0 or all metric mean_ci values are NaN
    n_cols = [f'{m}_n' for m in metrics if f'{m}_n' in summary_df.columns]
    mean_ci_cols = [f'{m}_mean_ci' for m in metrics if f'{m}_mean_ci' in summary_df.columns]

    # Identify rows to drop: any n==0 or all mean_ci are NaN
    drop_mask_n = summary_df[n_cols].eq(0).any(axis=1) if n_cols else pd.Series([False] * len(summary_df))
    drop_mask_mean_ci = summary_df[mean_ci_cols].isna().all(axis=1) if mean_ci_cols else pd.Series([False] * len(summary_df))

    drop_indices = summary_df.index[drop_mask_n | drop_mask_mean_ci]

    if len(drop_indices) > 0:
        print(f"Dropped {len(drop_indices)} groups with n=0 or all-NaN metrics")
        summary_df = summary_df.drop(drop_indices)

    if summary_df.empty:
        # Write placeholder to keep pipeline deterministic
        pd.DataFrame([{
            "note": "Summary table empty after drops.",
            "requested_groupby": " ".join(groupby),
            "requested_metrics": " ".join(metrics),
        }]).to_csv(out_path, index=False)
        print(f"Wrote placeholder CSV to {out_path}")
    else:
        save_table(summary_df, out_path, "csv")
        _write_markdown(summary_df, md_path)
        print(f"Saved summary table with {len(summary_df)} rows → CSV: {out_path} | MD: {md_path}")

if __name__ == "__main__":
    main()
