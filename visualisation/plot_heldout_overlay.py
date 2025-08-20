import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from visualisation.scenario import label_scenario
from visualisation.utils_config import find_run_dirs, load_config_with_fallback
from visualisation.style import setup_thesis_style, save_figure


def load_env_data(env_data_path):
    """Load environment metadata"""
    with open(env_data_path, "r") as f:
        return json.load(f)

def reconstruct_heldout_mask(config, grid_size):
    """Reconstruct held-out mask from config if not saved explicitly"""
    heldout_region = None

    # Check various possible config locations
    if "eval.heldout_region" in config:
        heldout_region = config["eval.heldout_region"]
    elif "eval" in config and "heldout_region" in config["eval"]:
        heldout_region = config["eval"]["heldout_region"]
    elif "heldout_region" in config:
        heldout_region = config["heldout_region"]

    if heldout_region is None:
        return None, None

    H, W = grid_size
    mask = np.zeros(H * W, dtype=bool)

    def in_region(i, j):
        if heldout_region == 'top_left':
            return i < H//2 and j < W//2
        elif heldout_region == 'top_right':
            return i < H//2 and j >= W//2
        elif heldout_region == 'bottom_left':
            return i >= H//2 and j < W//2
        elif heldout_region == 'bottom_right':
            return i >= H//2 and j >= W//2
        return False

    indices = []
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            if in_region(i, j):
                mask[idx] = True
                indices.append(idx)

    return mask, indices

def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s))[:200]

def plot_heldout_overlay(grid_size, heldout_indices, terminals, region_name, scenario_label, save_path):
    """Plot held-out region overlay"""
    setup_thesis_style()

    H, W = grid_size

    # Create base grid
    base_grid = np.zeros((H, W))

    # Mark held-out regions
    if heldout_indices:
        for idx in heldout_indices:
            i, j = divmod(idx, W)
            base_grid[i, j] = 1

    fig, ax = plt.subplots(figsize=(6, 5))
    extent = (-0.5, W - 0.5, H - 0.5, -0.5)  # explicit extents for correct alignment with grid lines

    # Plot base grid
    TRAIN_ALPHA = 0.3
    ax.imshow(np.ones((H, W)), cmap="gray", alpha=TRAIN_ALPHA, origin="upper", extent=extent, interpolation="nearest")

    # Overlay held-out region
    masked_grid = np.ma.masked_where(base_grid == 0, base_grid)
    OVERLAY_ALPHA = 0.35
    reds_cmap = plt.get_cmap("Reds")
    ax.imshow(masked_grid, cmap=reds_cmap, alpha=OVERLAY_ALPHA, origin="upper", extent=extent, interpolation="nearest")

    # Mark terminals
    for (i, j) in terminals:
        ax.text(j, i, '★', ha='center', va='center', color='blue', fontsize=14, weight='bold')

    # Add grid lines
    for i in range(H + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
    for j in range(W + 1):
        ax.axvline(j - 0.5, color='black', linewidth=0.5)

    # Axis ticks: integers 1..N, no tick marks; equal aspect and correct limits
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)  # origin='upper'
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.arange(W))
    ax.set_yticks(np.arange(H))
    ax.set_xticklabels([str(x) for x in range(1, W + 1)])
    ax.set_yticklabels([str(y) for y in range(1, H + 1)])
    ax.tick_params(axis='both', which='both', length=0)  # hide tick marks
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Enhanced title with scenario label
    title = f"Held-out Region: {region_name if region_name else 'None'}"
    if scenario_label:
        title += f" — {scenario_label}"
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Add legend
    held_rgba = list(reds_cmap(1.0))
    held_rgba[3] = OVERLAY_ALPHA  # enforce identical alpha
    train_rgba = (0.5, 0.5, 0.5, TRAIN_ALPHA)  # gray with same alpha as base layer
    legend_elements = [
        Patch(facecolor=held_rgba, edgecolor='none', label='Held-out (Test)'),
        Patch(facecolor=train_rgba, edgecolor='none', label='Training'),
        Line2D([0], [0], marker='*', color='blue', linestyle='None',
               markersize=10, label='Terminal States'),
    ]

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5))

    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close('all')

def plot_heldout_for_run(run_dir, out_dir):
    """Plot heldout overlay for a single run"""
    env_data_path = os.path.join(run_dir, "env_data.json")
    if not os.path.exists(env_data_path):
        print(f"Warning: env_data.json not found in {run_dir}, using default 5x5 grid")
        # Create default grid and continue processing
        grid_size = [5, 5]
        terminals = []
        heldout_indices = None
        region_name = None
    else:
        try:
            with open(env_data_path, 'r') as f:
                env_data = json.load(f)
            grid_size = env_data["grid_size"]
            terminals = env_data.get("terminal_states", [])

            # Try to get held-out info from env_data first
            heldout_indices = env_data.get("heldout_mask_indices")
            region_name = env_data.get("heldout_region")
        except Exception as e:
            print(f"Error reading env_data.json in {run_dir}: {e}, using defaults")
            grid_size = [5, 5]
            terminals = []
            heldout_indices = None
            region_name = None

    # Try to reconstruct held-out mask from config if not found
    try:
        if heldout_indices is None:
            config = load_config_with_fallback(run_dir)
            mask, indices = reconstruct_heldout_mask(config, grid_size)
            if indices is not None:
                heldout_indices = indices

                # Try to infer region name from config
                if region_name is None:
                    region_name = config.get("eval.heldout_region") or config.get("heldout_region")
    except Exception as e:
        print(f"Error processing config for {run_dir}: {e}")

    # Always ensure we have valid data for plotting
    if heldout_indices is None:
        heldout_indices = []
        if region_name is None:
            region_name = None

    # Get scenario label
    try:
        config = load_config_with_fallback(run_dir)
        scenario_label = label_scenario(config)
    except Exception:
        scenario_label = "unknown"

    # Always plot and save figure
    try:
        run_id = os.path.basename(run_dir.rstrip('/'))
        save_path = os.path.join(
            out_dir, f"{_sanitize(run_id)}__{_sanitize(scenario_label)}__heldout_overlay.png"
        )
        plot_heldout_overlay(grid_size, heldout_indices, terminals, region_name, scenario_label, save_path)

        # Print summary
        total_states = grid_size[0] * grid_size[1]
        heldout_count = len(heldout_indices) if heldout_indices else 0
        print(f"Processed {run_dir}: {heldout_count}/{total_states} states held out")

    except Exception as e:
        print(f"Error processing {run_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Plot held-out region overlay")
    parser.add_argument('--roots', nargs='+', required=True, help='Root directories with runs')
    parser.add_argument('--out', required=True, help='Output directory under results/figures/heldout')

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    run_dirs = find_run_dirs(args.roots)
    if not run_dirs:
        print("No run directories found")
        return

    for run_dir in run_dirs:
        plot_heldout_for_run(run_dir, args.out)

if __name__ == '__main__':
    main()
