import argparse
import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from typing import Optional, Tuple, Dict, List
from visualisation.utils_config import load_config_with_fallback, get, find_run_dirs
from visualisation.style import setup_thesis_style, save_figure
from visualisation.scenario import label_scenario


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s))[:200]

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
        # terminals may be under 'terminal_states'
        terminals = env.get("terminal_states", []) or env.get("terminals", [])
        # heldout mask may be precomputed; else from list of coords
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

def _overlay_context(ax: plt.Axes, overlay: Optional[Dict], variance_shape: Tuple[int, int],
                     show_terminals: bool = True, show_heldout: bool = True):
    """Draw terminals and held-out mask overlays if shape-compatible."""
    if not overlay:
        return
    (H, W) = variance_shape
    if overlay.get("grid_size") and tuple(overlay["grid_size"]) != (H, W):
        return
    if show_heldout and isinstance(overlay.get("heldout_mask"), np.ndarray):
        mask = overlay["heldout_mask"].astype(float)
        ax.imshow(mask, origin="upper", alpha=0.25)  # subtle overlay
    if show_terminals:
        terms = overlay.get("terminals") or []
        for t in terms:
            if isinstance(t, (list, tuple)) and len(t) == 2:
                i, j = int(t[0]), int(t[1])
                ax.plot(j, i, marker='*', markersize=10, markeredgewidth=1.5,
                        markeredgecolor='white', color='black')

def compute_reward_variance(reward_items, metric='var'):
    """Compute pixelwise variance across reward maps.
    Accepts a list of file paths OR numpy arrays."""
    if len(reward_items) == 0:
        raise ValueError("No reward files found")

    if len(reward_items) < 2:
        print(f"Warning: Only {len(reward_items)} map(s) found - variance will be minimal/zero")

    reward_maps = []
    for item in reward_items:
        if isinstance(item, str):
            reward_maps.append(np.load(item))
        else:
            reward_maps.append(np.asarray(item))
    
    stack = np.stack(reward_maps, axis=0)

    if metric == 'var':
        result = np.var(stack, axis=0)
    elif metric == 'std':
        result = np.std(stack, axis=0)
    elif metric == 'mean':
        result = np.mean(stack, axis=0)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    scalar_result = np.mean(result)
    return result, scalar_result

def load_per_run_data(run_dir, per_z_subdir='per_z'):
    """Load per-z reward maps from a single run directory"""
    per_z_dir = os.path.join(run_dir, per_z_subdir)
    # Prefer direct subdir; otherwise search recursively (handles parent/glob cases)
    reward_files = []
    if os.path.exists(per_z_dir):
        reward_files = sorted(glob.glob(os.path.join(per_z_dir, "reward_map_z*.npy")))
    else:
        # Recursive search for nested per_z directories
        reward_files = sorted(glob.glob(os.path.join(run_dir, "**", per_z_subdir, "reward_map_z*.npy"), recursive=True))
    return reward_files or None

def _get_reward_map_array(run_dir):
    """Load a single learned reward map from a run (array)."""
    map_path = os.path.join(run_dir, "learned_reward_map.npy")
    if os.path.exists(map_path):
        return np.load(map_path)
    # fallback: reshape learned_reward.npy via env_data.json
    vec_path = os.path.join(run_dir, "learned_reward.npy")
    env_path = os.path.join(run_dir, "env_data.json")
    if os.path.exists(vec_path) and os.path.exists(env_path):
        with open(env_path, "r") as f:
            env = json.load(f)
        H, W = env["grid_size"]
        return np.load(vec_path).reshape(H, W)
    return None

def load_across_runs_data(roots, groupby_key, per_z_subdir='per_z'):
    """Load and group per-z reward maps across multiple runs"""
    run_dirs = find_run_dirs(roots)

    grouped_items = {}
    skipped_count = 0

    for run_dir in run_dirs:
        # Load config to get groupby value
        try:
            config = load_config_with_fallback(run_dir)
            group_val = get(config, groupby_key)

            # Fallback chain when primary groupby key is None
            if group_val is None:
                # Try expert.confounder_value fallback
                group_val = get(config, 'expert.confounder_value')

            if group_val is None:
                # Try parsing run_dir name for patterns like trainz0, testz1
                import re
                basename = os.path.basename(run_dir)
                match = re.search(r'(train|test)z\d+', basename)
                if match:
                    group_val = match.group(0)

            if group_val is None:
                # Skip silently but count for summary
                skipped_count += 1
                continue

            group_val = str(group_val)

            # Load per-z files from this run
            per_z_files = load_per_run_data(run_dir, per_z_subdir)
            if per_z_files:
                grouped_items.setdefault(group_val, []).extend(per_z_files)
            else:
                arr = _get_reward_map_array(run_dir)
                if arr is not None:
                    grouped_items.setdefault(group_val, []).append(arr)

        except Exception as e:
            print(f"Warning: Failed to process {run_dir}: {e}")
            continue

    # Single concise summary instead of per-run warnings
    if skipped_count > 0:
        print(f"Skipped {skipped_count} runs with missing grouping key or pattern")

    return grouped_items

def plot_variance_heatmap(variance, title="Reward Variance", save_path=None, metric='var',
                          violations=None, overlay=None, overlay_terminals=True, overlay_heldout=True):
    """Plot variance heatmap with optional violation markers"""
    setup_thesis_style()

    plt.figure(figsize=(6.5, 5.2))

    cmap = "hot" if metric == 'var' else "viridis"
    ax = plt.gca()
    im = ax.imshow(variance, cmap=cmap, origin="upper")

    label = {"var": "Variance", "std": "Std Dev", "mean": "Mean"}[metric]
    cbar = plt.colorbar(im, pad=0.03, shrink=0.85)
    cbar.set_label(label)

    # Optional context overlays
    _overlay_context(ax, overlay, variance.shape, show_terminals=overlay_terminals, show_heldout=overlay_heldout)

    plt.title(f"{title} (Mean {label}: {np.mean(variance):.4f})")

    # Add violation markers if provided
    if violations:
        for i, (idx, mag) in enumerate(violations):
            if isinstance(idx, (list, tuple)) and len(idx) == 2:
                row, col = idx
            elif isinstance(idx, int):
                row, col = divmod(idx, variance.shape[1])
            else:
                continue
            ax.plot(col, row, 'wo', markersize=8, markeredgewidth=2, markeredgecolor='red')
            ax.text(col, row, str(i+1), ha='center', va='center', fontsize=8, weight='bold')

    ax.axis("off")

    # --- Grid aesthetics: extents, aspect, integer ticks, spines off ---
    H, W = int(variance.shape[0]), int(variance.shape[1])
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)  # origin='upper'
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.arange(W))
    ax.set_yticks(np.arange(H))
    ax.set_xticklabels([str(x) for x in range(1, W + 1)], fontsize=8)
    ax.set_yticklabels([str(y) for y in range(1, H + 1)], fontsize=8)
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    if save_path:
        save_figure(plt.gcf(), save_path)
    else:
        plt.show()
    plt.close('all')

def load_invariance_violations(run_dir):
    """Load top violations from invariance_analysis.json if present"""
    invariance_path = os.path.join(run_dir, 'invariance_analysis.json')
    if not os.path.exists(invariance_path):
        return None

    try:
        with open(invariance_path, 'r') as f:
            data = json.load(f)
        violations = data.get('top_violations', [])
        magnitudes = data.get('violation_magnitudes', [0] * len(violations))
        return list(zip(violations[:5], magnitudes[:5]))  # Top 5
    except Exception as e:
        print(f"Warning: Failed to load invariance analysis from {run_dir}: {e}")
        return None

def save_placeholder_image(save_path, message, title="Insufficient Data"):
    """Save a placeholder image when insufficient data is available"""
    setup_thesis_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12,
            wrap=True, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    save_figure(fig, save_path)
    print(f"Placeholder saved: {save_path}")
    return None, None

def _save_scalar_sidecars(base_path_no_ext: str, metric: str, mode: str, groupby: Optional[str], value: float,
                          n_maps: Optional[int] = None, group_counts: Optional[Dict[str, int]] = None):
    """Write scalar sidecars as JSON and CSV (keep .txt for backwards-compat if present).
    Adds n_maps, and (in across-runs) can persist per-group counts."""
    payload = {
        "metric": metric,
        "mode": mode,
        "groupby": groupby,
        "value": float(value),
    }
    if n_maps is not None:
        payload["n_maps"] = int(n_maps)
    if group_counts:
        payload["group_counts"] = {str(k): int(v) for k, v in group_counts.items()}
    with open(base_path_no_ext + ".json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(base_path_no_ext + ".csv", "w") as f:
        header = "metric,mode,groupby,value"
        if n_maps is not None:
            header += ",n_maps"
        f.write(header + "\n")
        row = f"{metric},{mode},{_sanitize(groupby) if groupby else ''},{value:.6f}"
        if n_maps is not None:
            row += f",{int(n_maps)}"
        f.write(row + "\n")

def _save_group_counts_sidecar(base_path_no_ext: str, group_counts: Dict[str, int]):
    """For across-runs: persist per-group n_maps counts as a separate CSV."""
    path = base_path_no_ext + "_group_counts.csv"
    with open(path, "w") as f:
        f.write("group_val,n_maps\n")
        for k, v in group_counts.items():
            f.write(f"{_sanitize(k)},{int(v)}\n")
    print(f"Saved group counts → {path}")

def per_run_mode(run_dir, per_z_subdir, metric, save_path, title, overlay_terminals=True, overlay_heldout=True):
    """Process single run directory (or parent/glob), collecting per-z maps recursively."""
    # Expand parent/glob to leaf run dirs and collect per-z maps
    reward_files = load_per_run_data(run_dir, per_z_subdir)

    if not reward_files:
        # Try expanding run_dir if it's a parent directory or glob pattern
        candidate_dirs = []
        if any(ch in str(run_dir) for ch in "*?[]"):
            for p in glob.glob(run_dir):
                if os.path.isdir(p):
                    candidate_dirs.append(p)
        elif os.path.isdir(run_dir):
            candidate_dirs.append(run_dir)

        if candidate_dirs:
            leaf_dirs = find_run_dirs(candidate_dirs)
            collected = []
            for leaf in leaf_dirs:
                files = load_per_run_data(leaf, per_z_subdir)
                if files:
                    collected.extend(files)
            if collected:
                reward_files = sorted(set(collected))

    if not reward_files:
        print(f"No per-z files found in {run_dir}/{per_z_subdir}")
        message = f"No per-z files found in\n{os.path.basename(run_dir)}/{per_z_subdir}"
        return save_placeholder_image(save_path, message, title)

    if len(reward_files) < 2:
        print(f"Warning: Only {len(reward_files)} per-z file(s) found - insufficient for variance")
        message = f"Insufficient maps ({len(reward_files)}<2) in\n{os.path.basename(run_dir)}"
        return save_placeholder_image(save_path, message, title)

    n_maps = len(reward_files)
    print(f"Found {n_maps} per-z reward files")
    variance, mean_var = compute_reward_variance(reward_files, metric)

    # Load violation markers if available
    violations = load_invariance_violations(run_dir)
    # Load overlay from this run (context)
    overlay = _load_env_overlay(run_dir)
    # Title enriched with mode
    title_enriched = f"{title} — mode=per_run"
    plot_variance_heatmap(variance, title_enriched, save_path, metric, violations,
        overlay=overlay, overlay_terminals=overlay_terminals, overlay_heldout=overlay_heldout
    )

    return variance, mean_var, n_maps

def across_runs_mode(roots, groupby_key, per_z_subdir, metric, save_path, base_title,
                     overlay_terminals=True, overlay_heldout=True):
    """Process multiple runs grouped by confounder value"""
    grouped_items = load_across_runs_data(roots, groupby_key, per_z_subdir)

    if not grouped_items:
        print(f"No grouped files found using groupby='{groupby_key}'")
        message = f"No grouped files found\nusing groupby='{groupby_key}'"
        return save_placeholder_image(save_path, message, f"Grouped by {groupby_key} — mode=across_runs")

    print(f"Found {len(grouped_items)} groups: {list(grouped_items.keys())}")
    try:
        counts = {k: (len(v) if hasattr(v, '__len__') else 0) for k, v in grouped_items.items()}
        print(f"[groups] counts={counts}")
    except Exception:
        pass

    # Aggregate all files across groups for variance computation
    all_items = []
    for group_items in grouped_items.values():
        all_items.extend(group_items)
    total_n_maps = len(all_items)

    if len(all_items) < 2:
        print(f"Warning: Only {len(all_items)} total maps - variance will be minimal")
        message = f"Insufficient total maps ({len(all_items)}<2)\ngrouped by {groupby_key}"
        return save_placeholder_image(save_path, message, f"Grouped by {groupby_key}")

    # Enhanced title with groupby information
    title = f"{base_title} — mode=across_runs (groupby={groupby_key})"
    group_vals = list(grouped_items.keys())

    if len(group_vals) <= 3:  # Show values if not too many
        title += f"\nValues: {', '.join(group_vals)}"

    variance, mean_var = compute_reward_variance(all_items, metric)
    # Try to locate an overlay from any run matching shape
    overlay = None
    try:
        cand_runs = find_run_dirs(roots)
        for rd in cand_runs:
            ov = _load_env_overlay(rd)
            if ov and tuple(ov.get("grid_size", ())) == tuple(variance.shape):
                overlay = ov
                break
    except Exception:
        pass
    plot_variance_heatmap(variance, title, save_path, metric, overlay=overlay,
        overlay_terminals=overlay_terminals, overlay_heldout=overlay_heldout
    )

    return variance, mean_var, total_n_maps, counts

def main():
    parser = argparse.ArgumentParser(description="Plot reward invariance heatmap")
    parser.add_argument("--mode", type=str, choices=['per_run', 'across_runs'], default='per_run')
    parser.add_argument("--run_dir", type=str, help="Single run directory (per_run mode)")
    parser.add_argument("--per_z_dir", type=str, default="per_z", help="Per-z subdirectory")
    parser.add_argument("--roots", nargs="+", help="Root directories (across_runs mode)")
    parser.add_argument("--groupby", type=str, default="env.confounder_value", help="Config key to group by")
    parser.add_argument("--metric", type=str, choices=['var', 'std', 'mean'], default='var', help="Aggregation across maps: var|std|mean")
    parser.add_argument("--out", type=str, required=True, help="Output directory under results/figures/invariance")
    parser.add_argument("--title", type=str, default="Reward Invariance Across Z")
    parser.add_argument("--overlay_terminals", action="store_true", default=True, help="Overlay terminal states")
    parser.add_argument("--no-overlay_terminals", dest="overlay_terminals", action="store_false")
    parser.add_argument("--overlay_heldout", action="store_true", default=True, help="Overlay held-out mask")
    parser.add_argument("--no-overlay_heldout", dest="overlay_heldout", action="store_false")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.out, exist_ok=True)

    if args.mode == 'per_run':
        if not args.run_dir:
            parser.error("--run_dir required for per_run mode")
        base = _sanitize(os.path.basename(str(args.run_dir)))
        if base in ("*", ".", ""):
            base = "per_run"
        save_path = os.path.join(args.out, f"{base}__invariance_{args.metric}.png")
        variance, mean_var, n_maps = per_run_mode(args.run_dir, args.per_z_dir, args.metric, save_path, args.title,
            overlay_terminals=args.overlay_terminals, overlay_heldout=args.overlay_heldout
        )
    elif args.mode == 'across_runs':
        if not args.roots:
            parser.error("--roots required for across_runs mode")
        save_path = os.path.join(args.out, f"invariance_across_runs_{_sanitize(args.groupby)}__{args.metric}.png")
        variance, mean_var, total_n_maps, group_counts = across_runs_mode(
            args.roots, args.groupby, args.per_z_dir, args.metric, save_path, args.title,
            overlay_terminals=args.overlay_terminals, overlay_heldout=args.overlay_heldout
        )

    if variance is not None:
        print(f"Mean {args.metric.title()}: {mean_var:.4f}")
        # Save scalar result (JSON/CSV sidecars, plus legacy .txt)
        if args.mode == 'per_run':
            base_no_ext = os.path.join(args.out, f"scalar_{args.metric}_per_run")
            legacy_txt = os.path.join(args.out, f"scalar_{args.metric}_per_run.txt")
            _save_scalar_sidecars(base_no_ext, args.metric, args.mode, None, mean_var, n_maps=n_maps)
            with open(legacy_txt, 'w') as f:
                f.write(f"{mean_var:.6f}\n")
            print(f"Saved scalar {args.metric} to {base_no_ext}.json/.csv and {legacy_txt}")
        else:
            base_no_ext = os.path.join(args.out, f"scalar_{args.metric}_across_runs_{_sanitize(args.groupby)}")
            legacy_txt = base_no_ext + ".txt"
            _save_scalar_sidecars(base_no_ext, args.metric, args.mode, args.groupby, mean_var,
                                  n_maps=total_n_maps, group_counts=group_counts)
            _save_group_counts_sidecar(base_no_ext, group_counts)
            with open(legacy_txt, 'w') as f:
                f.write(f"{mean_var:.6f}\n")
            print(f"Saved scalar {args.metric} to {base_no_ext}.json/.csv and {legacy_txt}")

if __name__ == '__main__':
    main()
