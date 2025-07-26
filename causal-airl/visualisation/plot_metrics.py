import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse


def plot_metric_vs_param(
    df: pd.DataFrame,
    metric: str,
    xparam: str,
    hue: str,
    save_path: str = None,
    title: str = None,
    y_lim=None
):
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x=xparam, y=metric, hue=hue, marker="o")
    plt.title(title or f"{metric} vs {xparam}")
    if y_lim:
        plt.ylim(y_lim)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[Saved] {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--xparam", type=str, required=True)
    parser.add_argument("--hue", type=str, default="method")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--ylim", type=str, default=None)  # e.g., "(0,1)"

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    y_lim = eval(args.ylim) if args.ylim else None

    plot_metric_vs_param(
        df,
        metric=args.metric,
        xparam=args.xparam,
        hue=args.hue,
        save_path=args.save_path,
        title=args.title,
        y_lim=y_lim
    )
