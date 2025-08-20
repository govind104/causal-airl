import os
import json
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from typing import Optional
from visualisation.utils_config import find_run_dirs, load_config_with_fallback
from visualisation.style import setup_thesis_style, save_figure
from visualisation.scenario import label_scenario


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s))[:200]

def _save_placeholder(run_dir, out_dir, scenario_label, reason):
    """Save a deterministic placeholder image explaining why a plot was skipped."""
    setup_thesis_style()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.5, 0.6, "Reward maps unavailable", ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.4, f"{reason}\n{scenario_label}", ha="center", va="center", fontsize=10)
    out_path = os.path.join(
        out_dir, f"{_sanitize(os.path.basename(run_dir))}__{_sanitize(scenario_label)}__reward_3panel_placeholder.png"
    )
    save_figure(fig, out_path)
    plt.close('all')

def load_and_reshape_reward(path: str, env_data_path: str):
    reward = np.load(path)
    if reward.ndim == 1:
        map_path = path.replace('.npy', '_map.npy')
        if os.path.exists(map_path):
            reward = np.load(map_path)
        else:
            with open(env_data_path, "r") as f:
                env_data = json.load(f)
            grid_size = env_data.get("grid_size")
            if not grid_size:
                raise ValueError(f"grid_size missing in {env_data_path}")
            reward = reward.reshape(grid_size)
    return reward

def load_env_data(env_data_path: str):
    with open(env_data_path, "r") as f:
        return json.load(f)

def normalize_terminals(terminals_raw):
    """Normalize terminal_states into List[Tuple[int,int]]"""
    terminals = []

    if isinstance(terminals_raw, (list, tuple)):
        # Handle list of pairs
        for t in terminals_raw:
            if (isinstance(t, (list, tuple)) and len(t) == 2):
                try:
                    i, j = int(t[0]), int(t[1])
                    terminals.append((i, j))
                except (ValueError, TypeError):
                    continue
    elif isinstance(terminals_raw, np.ndarray):
        # Handle ndarray of shape (N,2) or (2,N)
        arr = np.asarray(terminals_raw)
        if arr.ndim == 2:
            if arr.shape[1] == 2:  # Shape (N, 2)
                terminals = [(int(arr[i, 0]), int(arr[i, 1])) for i in range(arr.shape[0])]
            elif arr.shape[0] == 2:  # Shape (2, N)
                terminals = [(int(arr[0, i]), int(arr[1, i])) for i in range(arr.shape[1])]

    return terminals

def plot_three_panel_rewards(run_dir, out_dir, overlay_terminals: bool = True):
    """Create 3-panel figure: True reward, Learned reward, Difference"""
    setup_thesis_style()

    # Scenario label for titles/filenames
    try:
        cfg = load_config_with_fallback(run_dir)
        scenario_label = label_scenario(cfg) if cfg else os.path.basename(run_dir)
    except Exception:
        scenario_label = os.path.basename(run_dir)

    env_data_path = os.path.join(run_dir, 'env_data.json')
    learned_reward_path = os.path.join(run_dir, 'learned_reward.npy')
    true_reward_path = os.path.join(run_dir, 'true_reward.npy')

    # Check essential files exist (only learned reward is required)
    if not os.path.exists(learned_reward_path):
        print(f"Skipping {run_dir}: learned_reward.npy not found")
        _save_placeholder(run_dir, out_dir, scenario_label, "learned_reward.npy not found")
        return

    # Load env data once if available (used for fallback + overlays)
    env_data: Optional[dict] = None
    if os.path.exists(env_data_path):
        try:
            env_data = load_env_data(env_data_path)
        except Exception as e:
            print(f"Warning: failed to parse {env_data_path}: {e}")
            env_data = None

    # Check if true reward is available; if not, try fallbacks:
    # 1) env_data['true_reward']
    # 2) (optional) construct from terminal_states + reward_value
    true_reward_available = os.path.exists(true_reward_path)

    try:
        learned_reward = load_and_reshape_reward(learned_reward_path, env_data_path)

        true_reward = None
        if true_reward_available:
            true_reward = load_and_reshape_reward(true_reward_path, env_data_path)
        else:
            # Fallback 1: embedded true_reward in env_data.json
            if env_data is not None:
                grid_size = env_data.get("grid_size")
                if grid_size and isinstance(env_data.get("true_reward"), (list, tuple)):
                    try:
                        tr = np.array(env_data["true_reward"], dtype=float)
                        if tr.shape != tuple(grid_size):
                            tr = tr.reshape(grid_size)
                        true_reward = tr
                        # Persist for downstream reuse (best-effort)
                        try:
                            np.save(true_reward_path, true_reward.astype(np.float32))
                            print(f"[info] Reconstructed true_reward.npy from env_data.json for {run_dir}")
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"Warning: failed to reshape env_data['true_reward'] for {run_dir}: {e}")
                # Fallback 2 (optional): terminals + reward_value → sparse true reward
                if true_reward is None and grid_size:
                    # Accept both keys: 'terminal_states' (old) and 'terminals' (new)
                    terms = normalize_terminals(
                        env_data.get("terminal_states", []) or env_data.get("terminals", [])
                    )
                    if terms and ("reward_value" in env_data):
                        try:
                            H, W = int(grid_size[0]), int(grid_size[1])
                            tr = np.zeros((H, W), dtype=float)
                            rv = float(env_data.get("reward_value", 1.0))
                            for (i, j) in terms:
                                if 0 <= i < H and 0 <= j < W:
                                    tr[i, j] = rv
                            true_reward = tr
                            try:
                                np.save(true_reward_path, true_reward.astype(np.float32))
                                print(f"[info] Synthesised true_reward.npy from terminals+reward_value for {run_dir}")
                            except Exception:
                                pass
                        except Exception as e:
                            print(f"Warning: failed to synthesise true reward from terminals for {run_dir}: {e}")

        # Safely compute diff_reward only when true_reward exists
        diff_reward = None
        if true_reward is not None:
            diff_reward = None
            try:
                diff_reward = learned_reward - true_reward
            except Exception as e:
                print(f"Warning: Could not compute difference for {run_dir}: {e}")

        # Load environment data for terminals if available
        terminals = []
        if overlay_terminals and env_data is not None:
            # Accept both keys: 'terminal_states' (old) and 'terminals' (new)
            terminals = normalize_terminals(
                env_data.get('terminal_states', []) or env_data.get('terminals', [])
            )

        # Calculate symmetric vmin/vmax for consistent diverging colormap (consider learned and true if present)
        candidates = [np.abs(learned_reward).max()]
        if true_reward is not None:
            candidates.append(np.abs(true_reward).max())
        if diff_reward is not None:
            candidates.append(np.abs(diff_reward).max())
        max_abs = max(float(c) for c in candidates if np.isfinite(c))

        vmin, vmax = -max_abs, max_abs

        # Create 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # True reward panel
        if true_reward is not None:
            im0 = axes[0].imshow(true_reward, cmap='RdBu_r', origin='upper', vmin=vmin, vmax=vmax)
            axes[0].set_title('True Reward')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        else:
            axes[0].text(0.5, 0.5, 'True Reward\nNot Available', ha='center', va='center',
                         transform=axes[0].transAxes, fontsize=12, color='gray')
            axes[0].set_title('True Reward')

        # Learned reward panel with consistent diverging colormap
        im1 = axes[1].imshow(learned_reward, cmap='RdBu_r', origin='upper', vmin=vmin, vmax=vmax)
        axes[1].set_title('Learned Reward')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Difference panel with consistent diverging colormap
        if true_reward is not None and diff_reward is not None:
            im2 = axes[2].imshow(diff_reward, cmap='RdBu_r', origin='upper', vmin=vmin, vmax=vmax)
            axes[2].set_title('Difference (Learned - True)')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        else:
            axes[2].text(0.5, 0.5, 'Difference\nNot Available\n(no R_true)', ha='center', va='center',
                        transform=axes[2].transAxes, fontsize=12, color='gray')
            axes[2].set_title('Difference (Learned - True)')

        # Overlay terminal states (only on panels that exist visually)
        if overlay_terminals and terminals:
            overlay_axes = [axes[1], axes[2]] + ([axes[0]] if true_reward is not None else [])
            for (i, j) in terminals:
                for ax in overlay_axes:
                    ax.plot(j, i, marker='*', markersize=10, markeredgewidth=1.5,
                            markeredgecolor='white', color='black')

        # Ensure axes are off to remove ticks/frames
        for ax in axes:
            ax.axis('off')

        # Add missing true reward annotation to figure title and small text
        run_id = os.path.basename(run_dir.rstrip('/'))
        # Try to show method in title if available
        try:
            method = (cfg and (cfg.get('irl', {}) or {}).get('method')) or 'unknown'
        except Exception:
            method = 'unknown'
        if true_reward is None:
            fig.suptitle(f'{run_id} — {scenario_label} — {method} — Reward Maps (no R_true)', fontsize=14)
            # Add small annotation inside the figure
            fig.text(0.02, 0.02, '(no R_true available)', fontsize=8, color='red', ha='left', va='bottom')
        else:
            fig.suptitle(f'{run_id} — {scenario_label} — {method} — Reward Maps', fontsize=14)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space for suptitle

        save_path = os.path.join(
            out_dir, f"{_sanitize(run_id)}__{_sanitize(scenario_label)}__reward_3panel.png"
        )
        save_figure(fig, save_path)

        plt.close('all')
        print(f"[ok] Reward panels for {os.path.basename(run_dir)} → {save_path} "
              f"(true={'file' if true_reward_available else ('env' if true_reward is not None else 'missing')})")

    except Exception as e:
        print(f"Error processing {run_dir}: {e}")
        _save_placeholder(run_dir, out_dir, scenario_label, f"error: {e}")
    finally:
        plt.close('all')

def main():
    parser = argparse.ArgumentParser(description="Plot 3-panel reward maps")
    parser.add_argument('--roots', nargs='+', required=True, help='Root directories with runs')
    parser.add_argument('--out', required=True, help='Output directory under results/figures/heatmaps')
    parser.add_argument('--overlay_terminals', action='store_true', default=True, help='Overlay terminal states if available (default: on)')
    parser.add_argument('--no-overlay_terminals', dest='overlay_terminals', action='store_false')

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_dirs = find_run_dirs(args.roots)
    if not run_dirs:
        print("No run directories found")
        return

    for run_dir in run_dirs:
        plot_three_panel_rewards(run_dir, args.out, overlay_terminals=args.overlay_terminals)

if __name__ == '__main__':
    main()
