import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def moving_average(arr, window_size=5):
    if window_size <= 1:
        return arr
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

def plot_training_curve(
    x,
    ys_dict: dict,
    title: str = "Training Curve",
    xlabel: str = "Iteration",
    ylabel: str = "Loss",
    save_path: str = None
):
    plt.figure(figsize=(6, 4))
    for label, values in ys_dict.items():
        y_mean = np.mean(values, axis=0)
        y_std = np.std(values, axis=0)
        plt.plot(x, y_mean, label=label)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Saved] {save_path}")
    plt.close()

def load_training_log(run_dir):
    candidates = ["training_log.json", "training_log.npz"]
    for name in candidates:
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            if path.endswith(".json"):
                with open(path, "r") as f:
                    return json.load(f)
            elif path.endswith(".npz"):
                return dict(np.load(path))
    raise FileNotFoundError(f"No training log found in {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curves from one or more experiment runs")
    parser.add_argument("--log_root", type=str, required=True, help="Single run dir OR parent directory of multiple runs")
    parser.add_argument("--save_path", type=str, required=True, help="Where to save output plot")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title")
    parser.add_argument("--ylabel", type=str, default="Loss")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Comma-separated list of metric keys (default: auto-detect)")
    parser.add_argument("--smooth", type=int, default=0, help="Optional smoothing window (e.g., 5 for moving average)")

    args = parser.parse_args()

    # Determine whether this is a single run or a folder of multiple runs
    run_dirs = [args.log_root] if os.path.exists(os.path.join(args.log_root, "training_log.json")) \
        else sorted([os.path.join(args.log_root, d) for d in os.listdir(args.log_root)
                    if os.path.isdir(os.path.join(args.log_root, d))])

    if not run_dirs:
        raise FileNotFoundError(f"No training runs found under {args.log_root}")

    collected = {}
    for run_dir in run_dirs:
        try:
            log = load_training_log(run_dir)
            metrics = args.metrics.split(",") if args.metrics else list(log.keys())
            for k in metrics:
                v = log.get(k)
                if isinstance(v, (list, np.ndarray)) and len(v) > 1:
                    v = np.array(v)
                    if args.smooth > 1:
                        v = moving_average(v, args.smooth)
                    collected.setdefault(k, []).append(np.array(v))
        except Exception as e:
            print(f"[Warning] Skipping {run_dir} due to error: {e}")

    if not collected:
        raise RuntimeError("No valid training logs found.")

    min_len = min(min(len(seq) for seq in vlist) for vlist in collected.values())
    x = list(range(min_len))
    for k in collected:
        collected[k] = [arr[:min_len] for arr in collected[k]]

    # Auto-title
    title = args.title
    if not title:
        tag = os.path.basename(args.log_root.rstrip("/"))
        title = f"{tag.replace('_', ' ').title()} Training Curve"

    plot_training_curve(x, collected, title, "Iteration", args.ylabel, args.save_path)