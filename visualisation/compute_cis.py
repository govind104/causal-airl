import argparse
import json
import os
import ast
import glob
import math
import csv
import sys
import numpy as np
import pandas as pd

from typing import Dict, List, Any, Tuple, Optional
from experiments.stats import bootstrap_mean_ci, cohens_d
from visualisation.utils_config import load_config_with_fallback, get


# Metric direction: True => higher-is-better, False => lower-is-better
METRIC_DIRECTION = {
    # Core metrics
    "reward_correlation": True,
    "reward_corr_train": True,
    "reward_corr_test": True,
    "policy_agreement": True,
    "value_correlation": True,
    "value_difference": False,
    "wall_time_sec": False,
    "env_steps": False,

    # Cross-Z metrics
    "cross_z_from_1_to_0": True,
    "cross_z_from_0_to_1": True,

    # Reward statistics (higher = better for most)
    "reward_gini_abs": False,  # Lower Gini = more equal
    "reward_hist_entropy": True,
    "reward_std": True,
    "reward_skewness": True,  # Can be either direction
    "reward_range": True,
    "reward_sparsity": False,  # Lower sparsity = better
    "reward_mse": False,
    "reward_variance": True,

    # Diversity metrics
    "trajectory_entropy": True,
    "trajectory_overlap": False,  # Lower overlap = more diverse

    # Causal metrics
    "epoch_kl_raw": False,
    "epoch_kl_post": False,
    "epoch_inv_loss": False,
    "z_entropy_approx": True,
}

HEADLINE_DEFAULT = [
    "final_reward_correlation",
    "final_value_correlation",
    "final_policy_agreement",
    "final_reward_corr_test",
    "final_wall_time_sec",
    "final_env_steps",
    "final_cross_z_from_1_to_0",
    "final_cross_z_from_0_to_1",
    "final_trajectory_entropy",
    "final_reward_gini_abs"
]

def normalize_args(arg_list):
    """Convert comma or space separated args to list"""
    if len(arg_list) == 1 and ',' in arg_list[0]:
        return [x.strip() for x in arg_list[0].split(',') if x.strip()]
    return arg_list

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

def load_from_csv(csv_paths):
    """Load data from CSV files; if a CSV looks summarised, auto-fallback to detailed_results.csv."""
    dfs = []
    summarised_detected = False
    for csv_path in csv_paths:
        df = _try_load_csv(csv_path)
        if df is None:
            continue
        dfs.append(df)
        if _csv_looks_summarised(df):
            summarised_detected = True

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)
    if summarised_detected:
        # Prefer sibling detailed_results.csv next to any provided CSV; fallback to common location
        fallbacks = [os.path.join(os.path.dirname(p), "detailed_results.csv") for p in csv_paths]
        fallbacks.append(os.path.join("results", "validation", "detailed_results.csv"))
        for cand in fallbacks:
            ddf = _try_load_csv(cand)
            if ddf is not None:
                print(f"Detected summarised CSV; auto-falling back to raw rows in {cand}")
                return ddf
        print("Input CSV looks summarised (mean/lo/hi present), but no detailed_results.csv found.\n"
              "Please supply results/validation/detailed_results.csv or a raw per-run CSV.")
        return pd.DataFrame()

    return combined_df

def _coerce_listlike_to_scalar(x: Any) -> float | None:
    """
    Convert list-like values (actual lists/arrays or stringified lists) to a scalar by taking the mean.
    Returns float or None if not coercible.
    """
    try:
        # Already numeric?
        if is_number(x):
            return float(x)
        # Numpy array / list / tuple
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
            if arr.size == 0:
                return None
            return float(np.nanmean(arr))
        # Stringified list e.g. "[0.1, 0.2]" or "[0.1 0.2]"
        if isinstance(x, str) and x.strip():
            s = x.strip().replace(" ", ",") if ("[" in x and "," not in x) else x
            try:
                obj = ast.literal_eval(s)
                return _coerce_listlike_to_scalar(obj)
            except Exception:
                # try to parse single number string
                if is_number(x):
                    return float(x)
        return None
    except Exception:
        return None

def is_number(x: Any) -> bool:
    try:
        return isinstance(x, (int, float)) or (isinstance(x, str) and x.strip() and not math.isnan(float(x)))
    except Exception:
        return False

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def find_run_logs(roots: List[str], fname: str = "training_logs.json") -> List[str]:
    hits = []
    for root in roots:
        root = os.path.abspath(root)
        if os.path.isfile(root) and os.path.basename(root) == fname:
            hits.append(root)
        else:
            hits.extend(glob.glob(os.path.join(root, "**", fname), recursive=True))
    return sorted(set(hits))

def find_run_logs_with_fallback(roots: List[str]) -> List[str]:
    """Find training_logs.json, fallback to metrics.json if missing"""
    logs = find_run_logs(roots, "training_logs.json")
    if not logs:
        print("No training_logs.json found, trying metrics.json fallback")
        logs = find_run_logs(roots, "metrics.json")
    return logs

def resolve_output_path(path: Optional[str]) -> Optional[str]:
    """Resolve output path, defaulting to results/tables/ for bare filenames"""
    if path is None:
        return None
    if os.path.dirname(path) == '':
        path = os.path.join('results', 'tables', path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def find_manifest(run_dir: str) -> Optional[Dict[str, Any]]:
    mpath = os.path.join(run_dir, "manifest.json")
    if os.path.exists(mpath):
        try:
            return load_json(mpath)
        except Exception:
            return None
    return None

def find_manifest_or_config(run_dir: str) -> Dict[str, Any]:
    """
    Returns {'overrides': <flat dict>} from manifest if present,
    else flattens config as a stand-in overrides dict.
    """
    man = find_manifest(run_dir)
    if man and isinstance(man.get("overrides"), dict):
        return {"overrides": man["overrides"]}
    cfg = load_config_with_fallback(run_dir)
    return {"overrides": cfg}

def flatten_metrics_from_log(obj: Any) -> Dict[str, List[float]]:
    """
    Tolerate a few shapes:
      1) list[iter] of dict(metric -> scalar)
      2) dict(metric -> list[iter] of scalar)
      3) dict with 'metrics' key in either of the above forms
    returns dict(metric -> list[float]) aligned by iteration index.
    """
    if isinstance(obj, dict) and "metrics" in obj:
        obj = obj["metrics"]

    if isinstance(obj, list):
        # list of records
        acc: Dict[str, List[float]] = {}
        for rec in obj:
            if not isinstance(rec, dict):
                continue
            for k, v in rec.items():
                if is_number(v):
                    acc.setdefault(k, []).append(float(v))
        # pad ragged lists to same length with nan
        maxlen = max((len(v) for v in acc.values()), default=0)
        for k, v in acc.items():
            if len(v) < maxlen:
                v.extend([float("nan")] * (maxlen - len(v)))
        return acc

    if isinstance(obj, dict):
        # dict(metric -> list) or dict(metric -> scalars by iter)
        acc: Dict[str, List[float]] = {}
        for k, v in obj.items():
            if isinstance(v, list) and v and all(is_number(x) or x is None for x in v):
                acc[k] = [float(x) if is_number(x) else float("nan") for x in v]
            elif is_number(v):
                acc[k] = [float(v)]
        return acc

    # unknown shape
    return {}

def pick_value(ts: List[float], metric: str, agg: str) -> float:
    """choose a single score from a time series for a run."""
    arr = np.asarray(ts, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    if agg == "final":
        return float(arr[-1])
    if agg == "best":
        higher_better = METRIC_DIRECTION.get(metric, True)
        return float(np.nanmax(arr) if higher_better else np.nanmin(arr))
    # default fallback: final
    return float(arr[-1])

def fmt_mean_ci(mean: float, lo: float, hi: float, decimals: int = 3) -> str:
    if any(map(lambda z: z is None or np.isnan(z), [mean, lo, hi])):
        return "nan"
    return f"{mean:.{decimals}f}±[{lo:.{decimals}f},{hi:.{decimals}f}]"

def truncate_label(label: str, max_len: int = 30) -> str:
    """Truncate long labels for markdown display"""
    if len(str(label)) > max_len:
        return str(label)[:max_len-1] + "…"
    return str(label)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", help="CSV file paths to process")
    ap.add_argument("--roots", nargs="+", help="result directories to scan (for training logs)")
    ap.add_argument("--metrics", nargs="+", default=HEADLINE_DEFAULT,
                    help="metric names (comma or space separated)")
    ap.add_argument("--groupby", nargs="+", required=True,
                    help="group-by keys (comma or space separated)")
    ap.add_argument("--baseline", type=str, default="airl",
                    help="baseline group value for Cohen's d (default: 'airl')")
    ap.add_argument("--ci", type=int, default=95, help="bootstrap CI percentage (e.g., 90, 95)")
    ap.add_argument("--B", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=None, help="rng seed for bootstrap")
    ap.add_argument("--agg", type=str, default="final", choices=["final", "best"], help="how to reduce per-run time series")
    ap.add_argument("--out", type=str, default="cis_table.csv",
                    help="output CSV path; if no directory given, saves under results/tables/")
    ap.add_argument("--md", type=str, default=None,
                    help="optional markdown table output path; if no directory given, saves under results/tables/")
    args = ap.parse_args()

    # Normalize arguments
    wanted_metrics = normalize_args(args.metrics)
    group_keys = normalize_args(args.groupby)

    # Resolve output paths
    out_csv = resolve_output_path(args.out)
    out_md = resolve_output_path(args.md)

    # Prepare containers
    values: Dict[str, Dict[tuple, List[float]]] = {}
    group_order: List[tuple] = []

    # Load from CSV (preferred for final_* metrics) OR from logs
    if args.csv:
        df = load_from_csv(args.csv)
        if df.empty:
            print("No valid CSV data found after schema checks/fallbacks.")
            # Write placeholder CSV/MD (deterministic pipeline)
            if out_csv:
                os.makedirs(os.path.dirname(out_csv), exist_ok=True)
                with open(out_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["note"])
                    w.writerow(["No valid CSV data found after schema checks/fallbacks."])
                print(f"wrote placeholder {out_csv}")
            if out_md:
                with open(out_md, "w") as f:
                    f.write("# Bootstrap B=%d, CI=%d%%\n\n" % (args.B, args.ci))
                    f.write("_No valid CSV data found after schema checks/fallbacks._\n")
                print(f"wrote {out_md}")
            return

        # Flatten list-typed metric columns consistently before CI computation
        for metric in wanted_metrics:
            if metric in df.columns:
                before_nonnull = df[metric].notna().sum()
                df[metric] = df[metric].apply(_coerce_listlike_to_scalar)
                after_nonnull = df[metric].notna().sum()
                print(f"[flatten] {metric}: non-null {before_nonnull} → {after_nonnull}")

        # Ensure group-by columns exist; otherwise degrade gracefully to 'all'
        group_cols = [g for g in group_keys if g in df.columns] or ["all"]
        if group_cols == ["all"]:
            df["all"] = "all"

        # build values
        for metric in wanted_metrics:
            if metric not in df.columns:
                continue
            values[metric] = {}
            for group_vals, gdf in df.groupby(group_cols):
                if not isinstance(group_vals, tuple):
                    group_vals = (group_vals,)
                vec_all = gdf[metric]
                n_before = int(vec_all.shape[0])
                vec = vec_all.dropna().to_numpy(dtype=float)
                n_after = int(vec.size)
                print(f"[group-clean] metric={metric} group={group_vals} rows: {n_before} → {n_after}")
                if vec.size == 0:
                    continue
                values[metric][group_vals] = vec.tolist()
                if group_vals not in group_order:
                    group_order.append(group_vals)

    else:
        logs = find_run_logs_with_fallback(args.roots)
        if not logs:
            print("No training logs or metrics found under given roots.")
            # Placeholder outputs for determinism
            if out_csv:
                os.makedirs(os.path.dirname(out_csv), exist_ok=True)
                with open(out_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["Note"])
                    w.writerow(["No training logs or metrics found under given roots."])
                print(f"wrote placeholder {out_csv}")
            if out_md:
                with open(out_md, "w") as f:
                    f.write("# Bootstrap B=%d, CI=%d%%\n\n" % (args.B, args.ci))
                    f.write("_No training logs or metrics found under given roots._\n")
                print(f"wrote {out_md}")
            return

        # Process each run log independently
        for log_path in logs:
            run_dir = os.path.dirname(log_path)
            manifest = find_manifest_or_config(run_dir)
            ov = manifest.get("overrides", {})
            group_vals = []
            for k in group_keys:
                val = get(ov, k) if isinstance(ov, dict) else None
                group_vals.append("unspecified" if val is None else str(val))
            group = tuple(group_vals) if group_vals else ("all",)

            if group not in group_order:
                group_order.append(group)

            try:
                raw = load_json(log_path)
            except Exception as e:
                print(f"warn: failed to read {log_path}: {e}")
                continue

            # If metrics unspecified/empty, try to infer from present keys
            series = flatten_metrics_from_log(raw)
            present = set(series.keys())
            for metric in wanted_metrics:
                if metric not in present:
                    continue
                v = pick_value(series[metric], metric, agg=args.agg)
                values.setdefault(metric, {}).setdefault(group, []).append(v)

    if not values:
        print("No matching metrics found after processing.")
        # Write placeholders if requested
        if out_csv:
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Note"])
                w.writerow(["No matching metrics found after processing."])
            print(f"wrote placeholder {out_csv}")
        if out_md:
            with open(out_md, "w") as f:
                f.write("# Bootstrap B=%d, CI=%d%%\n\n" % (args.B, args.ci))
                f.write("_No matching metrics found after processing._\n")
            print(f"wrote {out_md}")
        return

    # If group order is still empty, derive from collected values
    if not group_order:
        for metric_dict in values.values():
            for g in metric_dict.keys():
                if g not in group_order:
                    group_order.append(g)

    # For baseline, use first group or parse baseline string
    baseline_group = None
    if args.baseline:
        # Try to find baseline in group_order
        for group in group_order:
            if isinstance(group, tuple) and args.baseline in group:
                baseline_group = group
                break
    if baseline_group is None:
        baseline_group = group_order[0] if group_order else None
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)

    for metric in wanted_metrics:
        if metric not in values:
            continue
        groups = values[metric]
        # ensure baseline appears first in output ordering
        ordered = [g for g in group_order if g in groups]
        if baseline_group and baseline_group in groups and baseline_group in ordered:
            # rotate so baseline is first
            while ordered and ordered[0] != baseline_group:
                ordered = ordered[1:] + ordered[:1]

        base_vec = np.asarray(groups[baseline_group], dtype=float) if baseline_group in groups else None
        for g in ordered:
            vec = np.asarray(groups[g], dtype=float)
            # clean nans
            vec = vec[~np.isnan(vec)]
            mean, lo, hi = bootstrap_mean_ci(vec, ci=args.ci, B=args.B, rng=rng)
            d = float("nan")
            if base_vec is not None and g != baseline_group:
                # compute d (this - baseline). sign indicates direction of difference.
                d = cohens_d(vec, base_vec)
            row = {
                "metric": metric,
                "n": int(vec.size),
                "mean": mean,
                f"ci{args.ci}_low": lo,
                f"ci{args.ci}_high": hi,
                "mean±ci": fmt_mean_ci(mean, lo, hi),
            }
            # Expand named group columns
            for i, k in enumerate(group_keys):
                row[k] = g[i] if len(g) > i else g[0] if g else "all"

            if baseline_group and g != baseline_group:
                row[f"d_vs_baseline"] = d
            elif baseline_group:
                row[f"d_vs_baseline"] = 0.0

            rows.append(row)

    # Write CSV
    if not group_keys:
        group_keys = ["group"]
    fieldnames = ["metric"] + group_keys + ["n", "mean", f"ci{args.ci}_low", f"ci{args.ci}_high", "mean±ci"]

    if baseline_group is not None:
        fieldnames.append("d_vs_baseline")
    with open(out_csv, "w", newline="") as f:
        # Header row that echoes group-by keys and CI setup
        meta_line = (
            f"# groupby={','.join(group_keys)} | "
            f"ci={args.ci}% | B={args.B} | agg={args.agg} | baseline={args.baseline}"
        )
        f.write(meta_line + "\n")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"wrote {out_csv} with {len(rows)} rows "
          f"(groupby={','.join(group_keys)}; CI={args.ci}%; B={args.B}; agg={args.agg}; baseline={args.baseline})")

    # Optional markdown
    if args.md:
        def format_num_md(val):
            """Format numbers for markdown, replacing NaN with em dash"""
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "—"
            return f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)

        # Simple grouped markdown
        with open(out_md, "w") as f:
            f.write(f"# Bootstrap B={args.B}, CI={args.ci}%\n\n")
            f.write(f"**Group-by:** {', '.join(group_keys)}  \n")
            f.write(f"**Baseline:** {baseline_group if baseline_group is not None else '—'}  \n\n")
            cur_metric = None
            for r in rows:
                m = r["metric"]
                if m != cur_metric:
                    if cur_metric is not None:
                        f.write("\n\n")
                    f.write(f"### {m}\n\n")
                    hdr = "| " + " | ".join(group_keys) + " | n | mean | ci_low | ci_high | mean±ci |"
                    if baseline_group is not None:
                        hdr += " d_vs_baseline |"
                    f.write(hdr + "\n")
                    sep_line = "|" + "---|" * len(group_keys) + "---:|---:|---:|---:|---:|" + ("---:|" if baseline_group is not None else "")
                    f.write(sep_line + "\n")
                    cur_metric = m

                # Truncate group values for markdown display only
                group_vals = []
                for k in group_keys:
                    val = str(r.get(k, ""))
                    group_vals.append(truncate_label(val))

                line = f"| {' | '.join(group_vals)} | {r['n']} | {format_num_md(r['mean'])} | {format_num_md(r[f'ci{args.ci}_low'])} | {format_num_md(r[f'ci{args.ci}_high'])} | {r.get('mean±ci', '—')} |"
                if baseline_group is not None:
                    dval = r.get("d_vs_baseline")
                    line += f" {('%.3f' % dval) if (dval is not None and not np.isnan(dval)) else '—'} |"
                f.write(line + "\n")
        print(f"wrote {out_md}")

if __name__ == "__main__":
    main()
