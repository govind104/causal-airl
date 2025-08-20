import os
import json
import glob
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from visualisation.utils_config import find_run_dirs, load_config_with_fallback
from visualisation.style import setup_thesis_style, save_figure
from visualisation.scenario import label_scenario


def moving_average(arr, window_size=5):
    if window_size <= 1:
        return arr
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

def find_checkpoints(run_dir):
    """Find checkpoint iterations from checkpoint files"""
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    if not os.path.isdir(ckpt_dir) or not os.listdir(ckpt_dir):
        print(f"No checkpoints found in {ckpt_dir}")
        return []
    files = glob.glob(os.path.join(ckpt_dir, 'reward_iter_*.npy'))
    if not files:
        print(f"No checkpoints found in {ckpt_dir}")
        return []

    iters = []
    for f in files:
        try:
            base = os.path.basename(f)
            num = int(base.split('_')[-1].split('.')[0])
            iters.append(num)
        except:
            continue
    return sorted(iters)

def filter_invalid_values(values, metric_name):
    """Filter out NaN and ±inf values with warning if any dropped."""
    filtered_vals = []
    dropped_count = 0

    for v in values:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            dropped_count += 1
        else:
            filtered_vals.append(v)

    if dropped_count > 0:
        print(f"Warning: Dropped {dropped_count} inf/NaN values in metric '{metric_name}'")

    return filtered_vals

def infer_method(run_dir):
    """Infer IRL method from config"""
    try:
        config = load_config_with_fallback(run_dir)
        method = config.get('irl.method') or config.get('method')
        if method:
            return method.lower()
    except Exception:
        pass
    return None

def get_metric_aliases(metric):
    """Define metric aliases for cross-method compatibility"""
    # Canonical targets:
    #   discriminator_loss  ← disc_bce | D_loss | disc_loss
    #   policy_loss         ← pi_loss  | actor_loss
    #   epoch_inv_loss      ← invariance_loss
    aliases = {
        'discriminator_loss': ['disc_bce', 'D_loss', 'disc_loss'],
        'disc_bce': ['discriminator_loss', 'D_loss', 'disc_loss'],
        'D_loss': ['discriminator_loss', 'disc_bce', 'disc_loss'],
        'policy_loss': ['pi_loss', 'actor_loss'],
        'pi_loss': ['policy_loss', 'actor_loss'],
        'epoch_inv_loss': ['invariance_loss'],
        'invariance_loss': ['epoch_inv_loss'],
    }
    return aliases.get(metric, [])

def find_metric_in_logs(logs, metric):
    """Find metric in logs, trying aliases if original not found.
    Returns (values, used_name, aliased_flag) or (None, None, False)."""
    if metric in logs:
        return logs[metric], metric, False
    for alias in get_metric_aliases(metric):
        if alias in logs:
            return logs[alias], alias, True
    return None, None, False

def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s))[:200]

def _save_placeholder(run_dir, out_dir, scenario_label, reason):
    """Save a deterministic placeholder when nothing is plottable for a run."""
    setup_thesis_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.text(0.5, 0.6, "No usable training curves", ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.38, reason, ha="center", va="center", fontsize=9, wrap=True)
    ax.text(0.5, 0.35, scenario_label, ha="center", va="center", fontsize=10, alpha=0.8)
    out_path = os.path.join(
        out_dir, f"{_sanitize(os.path.basename(run_dir))}__{_sanitize(scenario_label)}__training_placeholder.png"
    )
    save_figure(fig, out_path)
    plt.close('all')

def plot_training_curves_for_run(run_dir, logfile_name, metrics, smooth, out_dir):
    """Plot training curves for a single run"""
    setup_thesis_style()

    # Scenario label for context in titles/filenames
    try:
        cfg = load_config_with_fallback(run_dir)
        scenario_label = label_scenario(cfg) if cfg else os.path.basename(run_dir)
    except Exception:
        scenario_label = os.path.basename(run_dir)

    checked_files = [logfile_name, 'metrics.json']
    log_path = os.path.join(run_dir, logfile_name)
    if not os.path.isfile(log_path):
        # Fallback to metrics.json
        fallback_path = os.path.join(run_dir, 'metrics.json')
        if os.path.isfile(fallback_path):
            log_path = fallback_path
            print(f"Note: Using metrics.json fallback for {run_dir}")
        else:
            print(f"Skipping {run_dir}: neither {logfile_name} nor metrics.json found")
            _save_placeholder(
                run_dir, out_dir, scenario_label,
                f"Checked files: {', '.join(checked_files)} — none found"
            )
            return

    try:
        with open(log_path, 'r') as f:
            logs = json.load(f)
    except Exception as e:
        print(f"Skipping {run_dir}: failed to load log file ({e})")
        _save_placeholder(run_dir, out_dir, scenario_label, f"log parse error: {e}")
        return

    # If no metrics specified, use method-specific defaults
    if not metrics:
        method = infer_method(run_dir)
        if method == 'airl':
            metrics = ['discriminator_loss', 'policy_loss']
        elif method in ['causal-airl', 'causal_airl']:
            metrics = ['disc_bce', 'epoch_inv_loss', 'policy_loss']
        else:
            print(f"Warning: No metrics specified and method unknown for {run_dir}")
            _save_placeholder(
                run_dir, out_dir, scenario_label,
                f"Checked files: {os.path.basename(log_path)} (found); "
                f"unknown method & no metrics specified → nothing to plot"
            )
            return

    fig, ax = plt.subplots(figsize=(10, 6))
    max_len = 0
    plotted_any = False
    requested_metrics = list(metrics)
    found_metrics = []
    missing_metrics = []

    for metric in metrics:
        vals, used_name, aliased = find_metric_in_logs(logs, metric)
        if vals is None:
            # Only note missing if no aliases were found either
            missing_metrics.append(metric)
            continue

        if isinstance(vals, (list, np.ndarray)):
            arr = np.array(vals)
            if len(arr) == 0:
                continue

            # Filter out invalid values before processing
            filtered_vals = filter_invalid_values(arr.tolist(), used_name or metric)
            if len(filtered_vals) == 0:
                print(f"No valid data points for metric '{used_name or metric}' in {run_dir}")
                continue

            filtered_arr = np.array(filtered_vals)

            if smooth > 1 and len(arr) >= smooth:
                arr_smoothed = moving_average(filtered_arr, smooth)
                # Ensure x and y lengths match after smoothing
                x = np.arange(len(arr_smoothed))
                label = f"{used_name or metric}{' (alias)' if aliased else ''} (MA={smooth})"
                ax.plot(x, arr_smoothed, label=label)
                max_len = max(max_len, len(arr_smoothed))
            else:
                label = f"{used_name or metric}{' (alias)' if aliased else ''}"
                ax.plot(filtered_arr, label=label)
                max_len = max(max_len, len(filtered_arr))
            found_metrics.append(used_name or metric)
            plotted_any = True
        else:
            print(f"Metric {used_name or metric} is not a time series in {run_dir}")
            continue

    # If nothing was plotted, emit a placeholder and bail out
    if not plotted_any:
        reason = (
            f"Checked files: {os.path.basename(log_path)} (found). "
            f"Requested metrics: {', '.join(requested_metrics) if requested_metrics else '(none)'}; "
            f"found series: {', '.join(found_metrics) if found_metrics else '(none)'}; "
            f"missing: {', '.join(missing_metrics) if missing_metrics else '(none)'}; "
            f"no valid time-series after filtering."
        )
        _save_placeholder(run_dir, out_dir, scenario_label, reason)
        return

    # Add checkpoint vertical lines
    ckpt_iters = find_checkpoints(run_dir)
    for i, ckpt in enumerate(ckpt_iters):
        if ckpt < max_len:
            ax.axvline(ckpt, color='gray', linestyle='--', alpha=0.6,
                      label='Checkpoint' if i == 0 else None)

    # Ensure y-limits are recomputed after adding lines and leave a small margin
    ax.relim()
    ax.autoscale_view()
    ax.margins(y=0.05)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Metric Value')
    ax.set_title(f"Training Curves — {scenario_label}")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))

    out_path = os.path.join(
        out_dir, f"{_sanitize(os.path.basename(run_dir))}__{_sanitize(scenario_label)}__training.png"
    )
    save_figure(fig, out_path)

    print(f"[ok] Plotted {len(found_metrics)} metric(s) for {os.path.basename(run_dir)} → {out_path}")
    plt.close('all')

def plot_causal_training_curves(
    x,
    logs,
    title: str = "Causal Training Metrics",
    save_path: str = None,
    smooth: int = 0
):
    """Plot causal training metrics in separate panels"""
    causal_metrics = {
        'epoch_inv_loss': 'Invariance Loss',
        'epoch_kl_raw': 'KL Divergence (Raw)',
        'epoch_kl_post': 'KL Divergence (Clipped)'
    }

    # Filter available metrics
    available_metrics = {k: v for k, v in causal_metrics.items() if k in logs}

    if not available_metrics:
        print("[Warning] No causal metrics found in logs")
        return

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(8, 3 * n_metrics))

    if n_metrics == 1:
        axes = [axes]

    for i, (key, label) in enumerate(available_metrics.items()):
        values = logs[key]
        if isinstance(values, (list, np.ndarray)) and len(values) > 1:
            y = np.array(values)[:len(x)]
            if smooth > 1:
                y = moving_average(y, smooth)
                x_smooth = x[:len(y)]
            else:
                x_smooth = x

            axes[i].plot(x_smooth, y, label=label)
            axes[i].set_ylabel(label)
            axes[i].grid(True, linestyle="--", alpha=0.3)

            if i == len(available_metrics) - 1:
                axes[i].set_xlabel("Iteration")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        causal_save_path = save_path.replace('.pdf', '_causal.pdf').replace('.png', '_causal.png')
        os.makedirs(os.path.dirname(causal_save_path), exist_ok=True)
        plt.savefig(causal_save_path, dpi=300)
        print(f"[Saved] {causal_save_path}")
        plt.close()
    else:
        plt.show()

def load_training_log(run_dir):
    # Prefer plural; keep legacy names for backward-compatibility
    candidates = ["training_logs.json", "training_log.json", "training_log.npz"]
    for name in candidates:
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            if path.endswith(".json"):
                with open(path, "r") as f:
                    return json.load(f)
            elif path.endswith(".npz"):
                return dict(np.load(path))
    raise FileNotFoundError(f"no training log found in {run_dir} (looked for: {', '.join(candidates)})")

def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument('--roots', nargs='+', required=True, help='Root directories with runs')
    parser.add_argument('--logfile', default='training_logs.json', help='Training log filename')
    parser.add_argument('--metrics', nargs='*', default=[], help='Metrics to plot (auto-detected if not specified)')
    parser.add_argument('--ma', type=int, default=1, help='Moving average window size')
    parser.add_argument('--out', required=True, help='Output directory under results/figures/training')

    args = parser.parse_args()

    run_dirs = find_run_dirs(args.roots)
    if not run_dirs:
        print("No run directories found")
        return

    for run_dir in run_dirs:
        plot_training_curves_for_run(run_dir, args.logfile, args.metrics, args.ma, args.out)

if __name__ == '__main__':
    main()
