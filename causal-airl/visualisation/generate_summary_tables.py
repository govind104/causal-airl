import pandas as pd
import os
import argparse


def generate_table(
    df: pd.DataFrame,
    groupby: list,
    metrics: list,
    aggregation="mean",
    round_digits=3
):
    """
    Group and summarise evaluation metrics into a table.
    """
    grouped = df.groupby(groupby)[metrics]
    if aggregation == "mean":
        table = grouped.mean()
    elif aggregation == "median":
        table = grouped.median()
    else:
        raise ValueError("Unsupported aggregation.")

    return table.round(round_digits).reset_index()


def save_table(df: pd.DataFrame, save_path: str, fmt="csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if fmt == "csv":
        df.to_csv(save_path, index=False)
    elif fmt == "latex":
        with open(save_path, "w") as f:
            f.write(df.to_latex(index=False, float_format="%.3f"))
    else:
        raise ValueError("Unsupported format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--groupby", type=str, required=True)  # e.g. "method,demos"
    parser.add_argument("--metrics", type=str, required=True)  # e.g. "reward_corr,policy_agreement"
    parser.add_argument("--out", type=str, required=True)      # path to .csv or .tex
    parser.add_argument("--agg", type=str, default="mean")
    parser.add_argument("--round", type=int, default=3)

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    groupby = args.groupby.split(",")
    metrics = args.metrics.split(",")

    table = generate_table(df, groupby, metrics, aggregation=args.agg, round_digits=args.round)
    fmt = "latex" if args.out.endswith(".tex") else "csv"
    save_table(table, args.out, fmt=fmt)
    print(f"[Saved] Table → {args.out}")
