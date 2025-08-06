import os
import glob
import json
import argparse
import pandas as pd
import numpy as np


def bold_best_values_latex(df: pd.DataFrame, metric_cols: list) -> pd.DataFrame:
    def format_cell(val, is_best):
        return f"\\textbf{{{val:.3f}}}" if is_best else f"{val:.3f}"

    df_formatted = df.copy()
    for col in metric_cols:
        if col not in df.columns:
            continue
        col_vals = df[col]
        if pd.api.types.is_numeric_dtype(col_vals):
            max_val = col_vals.max()
            df_formatted[col] = [
                format_cell(v, np.isclose(v, max_val)) for v in col_vals
            ]
    return df_formatted

def generate_table(df: pd.DataFrame, groupby: list, metrics: list, aggregation="mean", round_digits=3):
    grouped = df.groupby(groupby)[metrics]
    if aggregation == "mean":
        table = grouped.mean()
    elif aggregation == "median":
        table = grouped.median()
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    return table.round(round_digits).reset_index()


def save_table(df: pd.DataFrame, save_path: str, fmt="csv", bold_best=False, metric_cols=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if fmt == "csv":
        df.to_csv(save_path, index=False)
    elif fmt == "latex":
        df_to_save = bold_best_values_latex(df, metric_cols) if bold_best else df
        with open(save_path, "w") as f:
            f.write(df_to_save.to_latex(index=False, escape=False))
    else:
        raise ValueError("Unsupported format.")

def load_experiment_records(results_dir):
    records = []
    for run_dir in sorted(glob.glob(os.path.join(results_dir, "*"))):
        config_path = os.path.join(run_dir, "config.json")
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(config_path) or not os.path.exists(metrics_path):
            continue

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"[Warning] Skipping {run_dir} due to error: {e}")
            continue

        record = {
            "method": config.get("method", "unknown").replace("_", "-").title(),
            "env": config.get("env_id", "unknown"),
            "run": os.path.basename(run_dir)
        }
        record.update(config)
        record.update(metrics)
        records.append(record)

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary tables from experiment runs")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Parent directory of run folders (e.g. results/airl_ablation/)")
    parser.add_argument("--groupby", type=str, required=True,
                        help="Group by keys (comma-separated, e.g. method,n_trajectories)")
    parser.add_argument("--metrics", type=str, required=True,
                        help="Metric keys to summarise (comma-separated)")
    parser.add_argument("--out", type=str, required=True,
                        help="Path to output .csv or .tex file")
    parser.add_argument("--agg", type=str, default="mean",
                        help="Aggregation function: mean or median")
    parser.add_argument("--round", type=int, default=3)
    parser.add_argument("--bold_best", action="store_true",
                        help="If set, bold the best values in LaTeX output")

    args = parser.parse_args()

    df = load_experiment_records(args.results_dir)
    groupby = args.groupby.split(",")
    metrics = args.metrics.split(",")

    missing = [m for m in metrics if m not in df.columns]
    if missing:
        print(f"[Warning] Missing metrics in dataframe: {missing}")
        metrics = [m for m in metrics if m in df.columns]

    if not metrics:
        raise RuntimeError("No valid metrics to summarise.")

    table = generate_table(df, groupby, metrics, aggregation=args.agg, round_digits=args.round)
    fmt = "latex" if args.out.endswith(".tex") else "csv"
    save_table(table, args.out, fmt=fmt, bold_best=args.bold_best, metric_cols=metrics)
    print(f"[Saved] Table → {args.out}")