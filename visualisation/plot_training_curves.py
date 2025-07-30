import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_training_curve(
    x,
    ys: dict,
    title: str = "Training Curve",
    xlabel: str = "Iteration",
    ylabel: str = "Loss",
    save_path: str = None
):
    plt.figure(figsize=(6, 4))
    for label, y in ys.items():
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[Saved] {save_path}")
    plt.close()


def load_log(path: str):
    if path.endswith(".json"):
        with open(path, "r") as f:
            log = json.load(f)
    elif path.endswith(".npz"):
        log = dict(np.load(path))
    else:
        raise ValueError("Unsupported format")
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--title", type=str, default="Training Curve")
    parser.add_argument("--ylabel", type=str, default="Loss")
    parser.add_argument("--metrics", type=str, default=None)  # comma-separated

    args = parser.parse_args()

    log = load_log(args.log_path)
    metrics = args.metrics.split(",") if args.metrics else list(log.keys())
    ys = {k: log[k] for k in metrics}
    x = list(range(len(next(iter(ys.values())))))

    plot_training_curve(x, ys, args.title, "Iteration", args.ylabel, args.save_path)
