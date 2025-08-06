import os
import json
import glob
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def load_ablation_data(results_dir, x_key, method_filter=None):
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
            print(f"[Error] Failed to load run {run_dir}: {e}")
            continue

        method = config.get("method", "unknown").replace("_", "-").title()
        if method_filter and method.lower() != method_filter.lower():
            continue

        if x_key not in config:
            print(f"[Warning] Missing {x_key} in config for {run_dir}")
            continue

        record = {
            "method": method,
            "run_id": os.path.basename(run_dir),
            x_key: config[x_key],
        }
        record.update(metrics)
        records.append(record)

    df = pd.DataFrame(records)
    return df


def plot_metric(df, x_key, metric, save_path):
    if metric not in df.columns:
        print(f"[Warning] Metric {metric} not found in DataFrame.")
        return

    # Plot mean ± std by group
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", font_scale=1.2)
    sns.lineplot(
        data=df,
        x=x_key,
        y=metric,
        hue="method",
        marker="o",
        err_style="band",
        estimator="mean",
        ci="sd"
    )
    plt.title(f"{metric.replace('_', ' ').title()} vs {x_key}")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel(x_key.replace("_", " ").title())
    plt.legend(title="Method")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[Saved] {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot AIRL ablation summaries")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to results/airl_ablation/")
    parser.add_argument("--x_key", type=str, required=True,
                        help="Which config parameter to plot on x-axis (e.g. gamma, reward_type, n_trajectories)")
    parser.add_argument("--metrics", nargs="+", default=["reward_correlation", "policy_agreement"],
                        help="Which metric keys to plot from metrics.json")
    parser.add_argument("--save_prefix", type=str, required=True,
                        help="Save path prefix (no extension)")
    parser.add_argument("--method_filter", type=str, default=None,
                        help="Optional: Only include runs matching method (e.g. AIRL or Causal-AIRL)")
    args = parser.parse_args()

    df = load_ablation_data(args.results_dir, args.x_key, args.method_filter)

    if df.empty:
        raise RuntimeError("No valid runs found with both config and metrics.")

    for metric in args.metrics:
        if metric not in df.columns:
            print(f"[Warning] Skipping missing metric: {metric}")
            continue
        save_path = f"{args.save_prefix}_{metric}.pdf"
        plot_metric(df, args.x_key, metric, save_path)