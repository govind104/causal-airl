import os
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from operator import itemgetter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from visualisation.utils_config import load_config_with_fallback, get, find_run_dirs
from visualisation.style import setup_thesis_style, save_figure
from visualisation.scenario import label_scenario


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s))[:200]

def create_placeholder_png(message, save_path):
    """Create placeholder PNG with error message when pairing/processing fails"""
    setup_thesis_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12, wrap=True)
    ax.axis('off')
    save_figure(fig, save_path)
    plt.close('all')

def load_reward(reward_path):
    reward = np.load(reward_path)
    return reward

def load_env_shape(env_data_path):
    with open(env_data_path, "r") as f:
        env_data = json.load(f)
    if "grid_size" in env_data:
        return tuple(env_data["grid_size"])
    else:
        raise ValueError("grid_size not found in env_data.json")

def load_terminal_states(env_data_path):
    with open(env_data_path, "r") as f:
        env_data = json.load(f)
    return env_data.get("terminal_states", [])

def compute_reward_difference(r1, r2, mode="abs"):
    # Reshape if needed
    if r1.ndim == 1 and r2.ndim == 1:
        pass  # both flat, ok
    elif r1.ndim == 2 and r2.ndim == 2:
        pass  # both 2D, ok
    elif r1.ndim == 1:
        r1 = r1.reshape(r2.shape)
    elif r2.ndim == 1:
        r2 = r2.reshape(r1.shape)

    if mode == "abs":
        diff = np.abs(r1 - r2)
    elif mode == "signed":
        diff = r1 - r2
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return diff

def _normalize_terminals(terminals_raw):
    out = []
    try:
        for t in terminals_raw:
            if isinstance(t, (list, tuple)) and len(t) == 2:
                i, j = int(t[0]), int(t[1])
                out.append((i, j))
    except Exception:
        pass
    return out

def _pick_least_busy_corner(diff2d):
    """Choose UL/UR/LL/LR based on lowest |diff| density to reduce overlap."""
    H, W = diff2d.shape
    h2, w2 = H // 2, W // 2
    q = [
        ("UL", np.abs(diff2d[:h2, :w2]).sum()),
        ("UR", np.abs(diff2d[:h2, w2:]).sum()),
        ("LL", np.abs(diff2d[h2:, :w2]).sum()),
        ("LR", np.abs(diff2d[h2:, w2:]).sum()),
    ]
    q_sorted = sorted(q, key=itemgetter(1))
    # Return both best corner and a fallback order (for collision resolution)
    return q_sorted[0][0], [c for c, _ in q_sorted]

def plot_diff_heatmap_with_histogram(diff, shape, terminals, title, save_path):
    setup_thesis_style()

    # Ensure diff is 2D for plotting
    if len(shape) == 2:
        diff = diff.reshape(shape)
    elif diff.ndim != 2:
        diff = diff.reshape(-1, int(np.sqrt(len(diff))))

    # Compute summary statistics for stats box
    mean_val = np.mean(diff)
    abs_mean_val = np.abs(mean_val)
    max_val = np.max(diff)
    abs_max_val = np.abs(max_val)

    # Create a two-panel figure: LEFT = stats+hist, RIGHT = heatmap (+ cbar on far right)
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.34, 0.66])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(left=0.06, right=0.92, top=0.92, bottom=0.12)

    # Symmetric color scale using RdBu_r diverging colormap
    abs_max = np.abs(diff).max()
    im = ax_main.imshow(diff, cmap='RdBu_r', origin='upper', vmin=-abs_max, vmax=abs_max)
    ax_main.set_title(title)

    # Terminal states overlay
    terms = _normalize_terminals(terminals)
    for (i, j) in terms:
        ax_main.plot(j, i, 'k*', markersize=10, markeredgewidth=1, markeredgecolor='white')

    ax_main.axis('off')

    # Slimmer colorbar on the far right of the heatmap
    cbar = fig.colorbar(im, ax=ax_main, shrink=0.8, pad=0.03)
    cbar.set_label('AIRL − Causal-AIRL')

    # LEFT PANEL: stats text + histogram (stacked)
    ax_left.set_title("Summary")
    # Stats text at top of left panel
    stats_text = (
        f"mean: {mean_val:.3f}\n"
        f"abs(mean): {abs_mean_val:.3f}\n"
        f"max: {max_val:.3f}\n"
        f"abs(max): {abs_max_val:.3f}"
    )
    ax_left.text(0.02, 0.98, stats_text, transform=ax_left.transAxes, fontsize=10,
                 va='top', ha='left', bbox=dict(boxstyle="round,pad=0.4",
                 facecolor="white", edgecolor="black", alpha=0.8))
    # Histogram inset occupying the bottom 60% of the left panel
    ax_hist = inset_axes(ax_left, width="100%", height="60%", loc="lower left", borderpad=0.2)
    ax_hist.hist(diff.flatten(), bins=20, alpha=0.7, color='gray', edgecolor='black')
    ax_hist.set_xlabel('Difference', fontsize=9)
    ax_hist.set_ylabel('Count', fontsize=9)
    ax_hist.tick_params(labelsize=8)
    ax_hist.grid(True, alpha=0.3)
    # Clean up left panel spines/ticks (only histogram shows ticks)
    for spine in ax_left.spines.values():
        spine.set_visible(False)
    ax_left.set_xticks([])
    ax_left.set_yticks([])

    # Save without tight_layout to avoid global style warning
    save_figure(fig, save_path, tight=False)
    plt.close('all')

def find_runs_by_scenario_and_method(roots):
    """Group runs by scenario and method for pairing"""
    runs = {}  # scenario -> method -> run_dir

    run_dirs = find_run_dirs(roots)

    for run_dir in run_dirs:
        try:
            cfg = load_config_with_fallback(run_dir)
            scenario = label_scenario(cfg)
            method_raw = (get(cfg, 'irl.method') or 'unknown').lower().replace('-', '_')
            if 'causal' in method_raw:
                method = 'causal_airl'
            elif 'airl' in method_raw:
                method = 'airl'
            else:
                method = 'unknown'

            if scenario not in runs:
                runs[scenario] = {}
            runs[scenario][method] = run_dir

        except Exception as e:
            print(f"Skipping {run_dir}: {e}")
            continue

    return runs

def plot_airl_vs_causal_diff(airl_dir, causal_dir, out_dir):
    """Plot AIRL vs Causal AIRL reward difference"""

    # Get scenario from AIRL run config for consistent labeling
    cfg_flat = load_config_with_fallback(airl_dir)
    scenario = label_scenario(cfg_flat)

    env1_path = os.path.join(airl_dir, 'env_data.json')

    try:
        shape = load_env_shape(env1_path)
        terminals = load_terminal_states(env1_path)
        # Normalize terminals for safety
        terminals = _normalize_terminals(terminals)

        # Load rewards (try different possible names)
        airl_reward_paths = [os.path.join(airl_dir, f) for f in ['learned_reward.npy', 'reward.npy']]
        causal_reward_paths = [os.path.join(causal_dir, f) for f in ['learned_reward.npy', 'causal_reward.npy']]

        airl_reward = None
        for p in airl_reward_paths:
            if os.path.exists(p):
                airl_reward = load_reward(p)
                break

        causal_reward = None
        for p in causal_reward_paths:
            if os.path.exists(p):
                causal_reward = load_reward(p)
                break

        if airl_reward is None or causal_reward is None:
            missing_side = "AIRL" if airl_reward is None else "Causal-AIRL"
            print(f"Missing reward files for scenario {scenario}: {missing_side}")
            safe_scenario = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in scenario.lower())
            save_path = os.path.join(out_dir, f"{safe_scenario}_airl_vs_causal_diff.png")
            create_placeholder_png(f"Missing reward files (scenario: {scenario})\nMissing: {missing_side}", save_path)
            return

        # Shape sanity check
        if airl_reward.shape != causal_reward.shape:
            print(f"Shape mismatch for scenario {scenario}: AIRL {airl_reward.shape} vs Causal {causal_reward.shape}")
            safe_scenario = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in scenario.lower())
            save_path = os.path.join(out_dir, f"{safe_scenario}_airl_vs_causal_diff.png")
            create_placeholder_png(f"Shape mismatch (scenario: {scenario})\nAIRL: {airl_reward.shape}\nCausal: {causal_reward.shape}", save_path)
            return

        diff = compute_reward_difference(airl_reward, causal_reward, mode='signed')

        # Include scenario in title and filename
        title = f"Reward difference (AIRL − Causal-AIRL) — {scenario}"
        save_path = os.path.join(out_dir, f"{_sanitize(scenario)}__airl_vs_causal_diff.png")

        plot_diff_heatmap_with_histogram(diff, shape, terminals, title, save_path)

    except Exception as e:
        print(f"Error processing scenario {scenario}: {e}")
        # Save placeholder on unexpected errors too
        save_path = os.path.join(out_dir, f"{_sanitize(scenario)}__airl_vs_causal_diff.png")
        create_placeholder_png(f"Error processing scenario {scenario}:\n{e}", save_path)
        plt.close('all')

def main():
    parser = argparse.ArgumentParser(description="Plot reward differences between AIRL and Causal AIRL")
    parser.add_argument('--roots', nargs='+', required=True, help='Root directories with runs')
    parser.add_argument('--out', required=True, help='Output directory under results/figures/diff')

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    runs_by_scenario = find_runs_by_scenario_and_method(args.roots)

    for scenario, methods in runs_by_scenario.items():
        if 'airl' in methods and 'causal_airl' in methods:
            plot_airl_vs_causal_diff(methods['airl'], methods['causal_airl'], args.out)
        else:
            available = list(methods.keys())
            if len(available) == 1:
                print(f"Missing counterpart for pairing in scenario {scenario} (found: {available[0]})")
                safe_scenario = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in scenario.lower())
                save_path = os.path.join(args.out, f"{_sanitize(scenario)}__airl_vs_causal_diff.png")
                create_placeholder_png(f"Missing counterpart for pairing (found: {available})", save_path)
            # else: no runs for this scenario, skip silently

if __name__ == '__main__':
    main()
