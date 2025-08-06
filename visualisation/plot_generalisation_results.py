import os
import json
import glob
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def load_generalisation_data(results_dir, group_by="z_shift", method_filter=None):
    records = []
    for run_dir in glob.glob(os.path.join(results_dir, "*")):
        config_path = os.path.join(run_dir, "config.json")
        metrics_path = os.path.join(run_dir, "metrics.json")

        if not os.path.exists(config_path) or not os.path.exists(metrics_path):
            continue

        with open(config_path, "r") as f:
            config = json.load(f)
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        method = config.get("method", "unknown").replace("_", "-").title()

        if method_filter and method.lower() != method_filter.lower():
            continue

        # Form group label (e.g., z0→z1)
        train_z = config.get("train_z")
        test_z = config.get("test_z")
        if train_z is not None and test_z is not None:
            group = f"z{train_z}→z{test_z}"
        else:
            group = "unknown"

        records.append({
            "method": method,
            group_by: group,
            **metrics
        })

    return pd.DataFrame(records)


def plot_generalisation(df, x_key, metric, save_path):
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", font_scale=1.2)
    sns.barplot(data=df, x=x_key, y=metric, hue="method", ci="sd", capsize=0.1)
    plt.title(f"{metric.replace('_', ' ').title()} by {x_key}")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel(x_key)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[Saved] {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot generalisation results from IRL experiments")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to results/generalization/")
    parser.add_argument("--metrics", nargs="+", default=["reward_correlation", "policy_agreement"],
                        help="Which metric keys to plot from metrics.json")
    parser.add_argument("--save_prefix", type=str, required=True,
                        help="Save path prefix (no extension)")
    parser.add_argument("--x_key", type=str, default="z_shift",
                        help="Label for x-axis grouping (default: z_shift)")
    parser.add_argument("--method_filter", type=str, default=None,
                        help="Optional: Only include runs from this method (e.g. AIRL, Causal-AIRL)")
    args = parser.parse_args()

    df = load_generalisation_data(args.results_dir, group_by=args.x_key, method_filter=args.method_filter)

    if df.empty:
        raise RuntimeError("No valid generalisation runs found.")

    # Enforce consistent z_shift label ordering
    if "z_shift" in df.columns:
        canonical_order = ["z0→z0", "z0→z1", "z1→z0", "z1→z1"]
        df["z_shift"] = pd.Categorical(df["z_shift"], categories=canonical_order, ordered=True)
        df = df.sort_values("z_shift")

    for metric in args.metrics:
        if metric not in df.columns:
            print(f"[Warning] Skipping missing metric: {metric}")
            continue
        save_path = f"{args.save_prefix}_{metric}.pdf"
        plot_generalisation(df, args.x_key, metric, save_path)