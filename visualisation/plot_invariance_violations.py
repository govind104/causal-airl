import argparse
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from typing import Optional, Dict, Tuple
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from visualisation.utils_config import find_run_dirs, flatten_config, get
from visualisation.style import setup_thesis_style, save_figure
from visualisation.scenario import label_scenario


def _load_cfg_flat(run_dir):
    """Load flattened config dict, with fallbacks."""
    cfg = {}
    for name in ("config_flat.json", "config.json"):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    raw = json.load(f)
                cfg = raw if name == "config_flat.json" else flatten_config(raw)
            except Exception:
                cfg = {}
            break
    return cfg

def _load_env_overlay(run_dir: str) -> Optional[Dict]:
    """
    Load overlay context from env_data.json for a single run.
    Returns dict with fields:
      - 'grid_size': (H, W)
      - 'terminals': list[(i,j)]
      - 'heldout_mask': np.ndarray bool shape (H, W) if available
    """
    env_path = os.path.join(run_dir, "env_data.json")
    if not os.path.exists(env_path):
        return None
    try:
        with open(env_path, "r") as f:
            env = json.load(f)
        H, W = env.get("grid_size", [None, None])
        if H is None or W is None:
            return None
        terminals = env.get("terminal_states", []) or env.get("terminals", [])
        heldout_mask = None
        if isinstance(env.get("heldout_mask"), (list, tuple)):
            heldout_mask = np.array(env["heldout_mask"], dtype=bool)
            if heldout_mask.shape != (H, W):
                heldout_mask = heldout_mask.reshape(H, W)
        else:
            held_states = env.get("heldout_states") or env.get("heldout_indices") or []
            if held_states:
                heldout_mask = np.zeros((H, W), dtype=bool)
                for yx in held_states:
                    if isinstance(yx, (list, tuple)) and len(yx) == 2:
                        i, j = int(yx[0]), int(yx[1])
                        if 0 <= i < H and 0 <= j < W:
                            heldout_mask[i, j] = True
        return {"grid_size": (H, W), "terminals": terminals, "heldout_mask": heldout_mask}
    except Exception as e:
        print(f"Warning: failed to parse env_data.json for {run_dir}: {e}")
        return None

def _overlay_context(ax: plt.Axes, overlay: Optional[Dict], shape: Tuple[int, int],
                     show_terminals: bool = True, show_heldout: bool = True):
    if not overlay:
        return
    if overlay.get("grid_size") and tuple(overlay["grid_size"]) != tuple(shape):
        return
    if show_heldout and isinstance(overlay.get("heldout_mask"), np.ndarray):
        ax.imshow(overlay["heldout_mask"].astype(float), origin="upper", alpha=0.25)
    if show_terminals:
        for t in (overlay.get("terminals") or []):
            if isinstance(t, (list, tuple)) and len(t) == 2:
                i, j = int(t[0]), int(t[1])
                ax.plot(j, i, marker='*', markersize=10, markeredgewidth=1.5,
                        markeredgecolor='white', color='black')

def _sanitize(s):
    """Filename-safe label."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s))[:200]

def _save_placeholder(run_dir, out_dir, scenario_label, reason):
    """Save a deterministic placeholder PNG for this run."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(
        0.5, 0.5,
        f"No data: {reason}\n{scenario_label}",
        ha="center", va="center", fontsize=12
    )
    run_id = os.path.basename(run_dir)
    out_path = os.path.join(
        out_dir, f"{_sanitize(run_id)}__{_sanitize(scenario_label)}__violations_placeholder.png"
    )
    save_figure(fig, out_path)

def _write_topk_csv(out_dir: str, run_id: str, scenario_label: str, grid_size, indices, values):
    path = os.path.join(out_dir, f"{_sanitize(run_id)}__{_sanitize(scenario_label)}__violations_topK.csv")
    with open(path, "w") as f:
        f.write("rank,state_idx,row,col,value\n")
        for rank, (idx, val) in enumerate(zip(indices, values), start=1):
            r, c = (idx if isinstance(idx, (list, tuple)) else divmod(int(idx), grid_size[1]))
            f.write(f"{rank},{int(idx) if not isinstance(idx,(list,tuple)) else ''},{int(r)},{int(c)},{float(val) if val is not None else ''}\n")
    print(f"[ok] Wrote Top-K CSV → {path}")

def _write_scalar_sidecars(out_dir: str, run_id: str, scenario_label: str, mean_val: float, std_val: float, n_maps: int):
    base = os.path.join(out_dir, f"{_sanitize(run_id)}__{_sanitize(scenario_label)}__stdmap_scalar")
    with open(base + ".json", "w") as f:
        json.dump({"metric": "stdmap", "mean": float(mean_val), "std": float(std_val), "n_maps": int(n_maps)}, f, indent=2)
    with open(base + ".csv", "w") as f:
        f.write("metric,mean,std,n_maps\nstdmap,%.6f,%.6f,%d\n" % (mean_val, std_val, int(n_maps)))
    print(f"[ok] Saved scalars (n_maps=%d) → %s.json/.csv" % (int(n_maps), base))

def _compute_stdmap_mean(perz_dir: str, grid_size) -> Optional[float]:
    """Compute mean of std-map across Z (silent diagnostic)."""
    try:
        if not os.path.isdir(perz_dir):
            return None
        z_maps = []
        for f in sorted(os.listdir(perz_dir)):
            if f.startswith("reward_map_z") and f.endswith(".npy"):
                arr = np.load(os.path.join(perz_dir, f))
                if arr.ndim == 1 and grid_size is not None:
                    arr = arr.reshape(grid_size)
                z_maps.append(arr)
        if len(z_maps) < 2:
            return None
        stack = np.stack(z_maps, axis=0)
        stdmap = stack.std(axis=0)
        return float(np.mean(stdmap))
    except Exception:
        return None

def plot_invariance_violations(roots, out_dir, K=10, overlay_terminals=True, overlay_heldout=True):
    """Plot invariance violation maps."""
    setup_thesis_style()
    run_dirs = find_run_dirs(roots)

    for run_dir in run_dirs:
        # Load scenario context up-front (used for titles/filenames/placeholders)
        cfg_flat = _load_cfg_flat(run_dir)
        scenario_label = label_scenario(cfg_flat) if cfg_flat else os.path.basename(run_dir)
        run_id = os.path.basename(run_dir)

        invariance_path = os.path.join(run_dir, 'invariance_analysis.json')

        # Try to load overlay (grid, terminals, held-out)
        overlay = _load_env_overlay(run_dir)
        grid_size_overlay = overlay.get("grid_size") if overlay else None

        # Default plotting base: learned reward; but if invariance is missing and per-Z exists,
        # we'll compute and plot the std-map instead (diagnosable fallback).

        invariance_data = None
        if os.path.isfile(invariance_path):
            try:
                with open(invariance_path) as f:
                    invariance_data = json.load(f)
            except Exception as e:
                print(f"Skipping {run_dir} (invariance analysis parse error: {e})")
                _save_placeholder(run_dir, out_dir, scenario_label, "invariance_analysis.json parse error")
                continue

        # Load reward map
        reward_map_paths = [
            os.path.join(run_dir, 'learned_reward_map.npy'),
            os.path.join(run_dir, 'learned_reward.npy')
        ]

        reward_map = None
        for path in reward_map_paths:
            if os.path.isfile(path):
                try:
                    reward_map = np.load(path)
                    break
                except Exception:
                    continue

        # Load environment data (for shape fallback)
        env_data_path = os.path.join(run_dir, 'env_data.json')
        grid_size = None
        if os.path.isfile(env_data_path):
            try:
                with open(env_data_path) as f:
                    env_data = json.load(f)
                grid_size = env_data.get('grid_size', None)
            except Exception:
                grid_size = None

        if reward_map is None and grid_size is None:
            print(f"No reward map or env grid size for {run_dir}")
            _save_placeholder(run_dir, out_dir, scenario_label, "no reward map/env_data grid_size")
            continue
        if grid_size is None and reward_map is not None:
            # infer square-ish if possible
            side = int(np.sqrt(reward_map.size))
            grid_size = [side, reward_map.size // side]

        # Reshape reward map if needed
        if reward_map is not None and reward_map.ndim == 1:
            reward_map = reward_map.reshape(grid_size)

        # Extract violation information with robust fallbacks
        violation_indices = None
        violation_magnitudes = None
        if isinstance(invariance_data, dict):
            # Support both schemas
            violation_indices = invariance_data.get('top_violations') or invariance_data.get('violation_states')
            violation_magnitudes = invariance_data.get('violation_magnitudes')

        # If missing or empty, compute from per-Z reward maps: pick top-K by std across z and PLOT std-map
        use_stdmap = False
        stdmap = None
        if not violation_indices:
            perz_dir = os.path.join(run_dir, "per_z")
            z_maps = []
            if os.path.isdir(perz_dir):
                for f in sorted(os.listdir(perz_dir)):
                    if f.startswith("reward_map_z") and f.endswith(".npy"):
                        arr = np.load(os.path.join(perz_dir, f))
                        if arr.ndim == 1:
                            arr = arr.reshape(grid_size)
                        z_maps.append(arr)
            if len(z_maps) >= 2:
                stack = np.stack(z_maps, axis=0)  # (Z,H,W)
                stdmap = stack.std(axis=0)        # (H,W) — invariance proxy
                flat_idx = np.argsort(stdmap.ravel())[::-1]

                # Ensure K respects environment size
                max_K = grid_size[0] * grid_size[1]
                K_eff = min(int(K), max_K)

                violation_indices = flat_idx[:K_eff].tolist()
                violation_magnitudes = stdmap.ravel()[violation_indices].astype(float).tolist()
                use_stdmap = True
            else:
                # Consolidated "no data" message + deterministic placeholder
                print(f"No invariance analysis or sufficient per-z maps found for {run_dir}")
                _save_placeholder(run_dir, out_dir, scenario_label, "no invariance analysis or per-z (>=2) maps")
                continue

        # Determine K for titles/legend
        K_title = len(violation_indices) if violation_indices is not None else 0

        # Normalize indices to (r,c) and magnitudes to floats
        idx_rc = []
        mags = []
        for i, idx in enumerate(violation_indices):
            if isinstance(idx, (list, tuple)) and len(idx) == 2:
                r, c = int(idx[0]), int(idx[1])
            else:
                r, c = divmod(int(idx), grid_size[1])
            idx_rc.append((r, c))
            if violation_magnitudes is not None and i < len(violation_magnitudes):
                mags.append(float(violation_magnitudes[i]))
            else:
                mags.append(float('nan'))

        # Create figure
        fig, (ax_main, ax_table) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={'width_ratios': [2, 1]})

        # Plot base: stdmap (fallback) or learned reward
        if use_stdmap and stdmap is not None:
            im = ax_main.imshow(stdmap, cmap='hot', origin='upper')
            mean_std = float(np.mean(stdmap))
            std_std = float(np.std(stdmap))
            title = f'{scenario_label} — STD-map across Z (Top-K={K_title}) | mean={mean_std:.4f}, sd={std_std:.4f}'
            ax_main.set_title(title)
            plt.colorbar(im, ax=ax_main, shrink=0.8, pad=0.03).set_label("Std across Z")
            # Persist n_maps (number of Z slices used)
            try:
                # reconstruct number of maps from stdmap construction:
                perz_dir = os.path.join(run_dir, "per_z")
                z_files = [f for f in sorted(os.listdir(perz_dir)) if f.startswith("reward_map_z") and f.endswith(".npy")]
                n_maps = len(z_files)
            except Exception:
                n_maps = 0
            _write_scalar_sidecars(out_dir, run_id, scenario_label, mean_std, std_std, n_maps=n_maps)
        else:
            im = ax_main.imshow(reward_map, cmap='RdBu_r', origin='upper')
            ax_main.set_title(f'{scenario_label} — Learned Reward with Invariance Violations (Top-K={K_title})')
            plt.colorbar(im, ax=ax_main, shrink=0.8, pad=0.03)
            # Even when using invariance_analysis.json, silently compute std-map mean if possible
            mean_std_diag = _compute_stdmap_mean(os.path.join(run_dir, "per_z"), grid_size)
            if mean_std_diag is not None:
                print(f"[diag] stdmap_mean={mean_std_diag:.4f} (run={run_id})")

        # Overlays (terminals/held-out)
        _overlay_context(ax_main, overlay, (grid_size[0], grid_size[1]),
                         show_terminals=overlay_terminals, show_heldout=overlay_heldout)

        # Mark violation states
        for i, (idx, mag) in enumerate(zip(violation_indices, violation_magnitudes)):
            if isinstance(idx, list) and len(idx) == 2:
                row, col = idx
            elif isinstance(idx, (int, np.integer)):
                # Convert flat index to 2D
                row, col = divmod(idx, grid_size[1])
            else:
                print(f"Skipping invalid index format: {idx}")
                continue

            # Add marker
            circle = plt.Circle((col, row), 0.3, color='yellow', alpha=0.8, linewidth=2, fill=False)
            ax_main.add_patch(circle)
            ax_main.text(col, row, str(i+1), ha='center', va='center',
                        fontweight='bold', fontsize=8, color='black',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))

        ax_main.set_xlabel('Column')
        ax_main.set_ylabel('Row')
        # --- Subtle grid aesthetics & ticks ---
        H, W = int(grid_size[0]), int(grid_size[1])
        ax_main.set_xlim(-0.5, W - 0.5)
        ax_main.set_ylim(H - 0.5, -0.5)  # origin='upper'
        ax_main.set_aspect('equal', adjustable='box')
        ax_main.set_xticks(np.arange(W))
        ax_main.set_yticks(np.arange(H))
        ax_main.set_xticklabels([str(x) for x in range(1, W + 1)], fontsize=8)
        ax_main.set_yticklabels([str(y) for y in range(1, H + 1)], fontsize=8)
        ax_main.tick_params(axis='both', which='both', length=0)
        for spine in ax_main.spines.values():
            spine.set_visible(False)

        # Legend entry for Top-K markers
        legend_elements = [
            Line2D([0], [0], marker='o', color='yellow', markerfacecolor='none',
                   linestyle='None', markersize=8, label=f'Top-K invariance violations (K={K_title})')
        ]
        ax_main.legend(handles=legend_elements, loc='best', framealpha=0.9)

        # Create violations table
        ax_table.axis('off')
        table_data = []
        headers = ['Rank', 'State', 'Magnitude']

        for i, (idx, mag) in enumerate(zip(violation_indices[:10], violation_magnitudes[:10])):  # Top 10
            if isinstance(idx, list) and len(idx) == 2:
                state_str = f"({idx[0]}, {idx[1]})"
            elif isinstance(idx, (int, np.integer)):
                row, col = divmod(idx, grid_size[1])
                state_str = f"({row}, {col})"
            else:
                state_str = "unknown"
            table_data.append([f"{i+1}", state_str, f"{mag:.3f}"])

        table = ax_table.table(cellText=table_data, colLabels=headers,
                              cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)

        # Style table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax_table.set_title(f'Top Invariance Violations (Top-K={K_title})')

        # Generate output path
        out_path = os.path.join(
            out_dir, f'{_sanitize(run_id)}__{_sanitize(scenario_label)}__violations.png'
        )
        save_figure(fig, out_path)
        # Write Top-K CSV (state_idx, row, col, value)
        _write_topk_csv(out_dir, run_id, scenario_label, grid_size, violation_indices, violation_magnitudes)
        print(f"[ok] Invariance plot saved → {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot invariance violations with std-map fallback and overlays")
    parser.add_argument('--roots', nargs='+', required=True, help='Root directories with runs')
    parser.add_argument('--out', required=True, help='Output directory (auto-created)')
    parser.add_argument('--K', type=int, default=10, help='Top-K violating states to mark (default 10)')
    parser.add_argument('--overlay_terminals', action='store_true', default=True, help='Overlay terminal states')
    parser.add_argument('--no-overlay_terminals', dest='overlay_terminals', action='store_false')
    parser.add_argument('--overlay_heldout', action='store_true', default=True, help='Overlay held-out mask')
    parser.add_argument('--no-overlay_heldout', dest='overlay_heldout', action='store_false')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    plot_invariance_violations(args.roots, args.out, K=args.K,
                               overlay_terminals=args.overlay_terminals,
                               overlay_heldout=args.overlay_heldout)

if __name__ == '__main__':
    main()
