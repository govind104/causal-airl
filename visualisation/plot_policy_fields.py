import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from models.policy import PolicyNet
from visualisation.utils_config import find_run_dirs, load_config_with_fallback
from visualisation.style import setup_thesis_style, save_figure
from visualisation.scenario import label_scenario


ACTION_TO_DELTA = {
    0: (-1, 0),  # up
    1: ( 1, 0),  # down
    2: ( 0,-1),  # left
    3: ( 0, 1),  # right
}

def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s))[:200]

def _save_placeholder(run_dir, out_dir, scenario_label, reason):
    """Save a deterministic placeholder when policy field cannot be produced."""
    setup_thesis_style()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.5, 0.6, "Policy field unavailable", ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.4, f"{reason}\n{scenario_label}", ha="center", va="center", fontsize=10)
    out_path = os.path.join(
        out_dir, f"{_sanitize(os.path.basename(run_dir))}__{_sanitize(scenario_label)}__policy_placeholder.png"
    )
    save_figure(fig, out_path)
    plt.close('all')

def _normalize_terminals(terminals_raw):
    """Normalize terminals into List[(i,j)] with 0-based indices."""
    out = []
    if terminals_raw is None:
        return out
    # List/tuple of pairs
    if isinstance(terminals_raw, (list, tuple)):
        for t in terminals_raw:
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                try:
                    out.append((int(t[0]), int(t[1])))
                except (TypeError, ValueError):
                    continue
        return out
    # Numpy array (N,2) or (2,N)
    if isinstance(terminals_raw, np.ndarray) and terminals_raw.ndim == 2:
        if terminals_raw.shape[1] == 2:
            for i in range(terminals_raw.shape[0]):
                out.append((int(terminals_raw[i, 0]), int(terminals_raw[i, 1])))
        elif terminals_raw.shape[0] == 2:
            for j in range(terminals_raw.shape[1]):
                out.append((int(terminals_raw[0, j]), int(terminals_raw[1, j])))
    return out

def load_policy(run_dir, device="cpu"):
    """Load policy from policy.pt file"""
    policy_path = os.path.join(run_dir, 'policy.pt')
    annotation = None

    if not os.path.exists(policy_path):
        raise RuntimeError(f"policy.pt not found in {run_dir}")

    try:
        policy = torch.load(policy_path, map_location=device)

        # Check if policy is a state_dict rather than a callable model
        if isinstance(policy, dict) and all(isinstance(k, str) for k in policy.keys()):
            # Attempt policy reconstruction from state_dict
            env_data_path = os.path.join(run_dir, 'env_data.json')
            if os.path.exists(env_data_path):
                try:
                    with open(env_data_path, 'r') as f:
                        env_data = json.load(f)
                    grid_size = env_data.get('grid_size') or env_data.get('grid_shape')
                    if grid_size and len(grid_size) == 2:
                        state_dim = grid_size[0] * grid_size[1]  # one-hot state representation
                        action_dim = len(ACTION_TO_DELTA)

                        # Reconstruct PolicyNet from state_dict
                        reconstructed_policy = PolicyNet(state_dim, action_dim)
                        reconstructed_policy.load_state_dict(policy)
                        reconstructed_policy.eval()
                        return reconstructed_policy, None
                except Exception as e:
                    print(f"Warning: Policy reconstruction failed for {run_dir}: {e}")
                    annotation = "Policy reconstruction failed — fallback arrows"
            else:
                annotation = "env_data.json missing for policy reconstruction"
            # Fall back to returning None with annotation for fallback arrows
            return None, annotation

        if hasattr(policy, 'eval'):
            policy.eval()
        return policy, annotation

    except Exception as e:
        raise RuntimeError(f"Failed to load policy from {policy_path}: {e}")


def plot_vector_field(policy, grid_size, save_path, terminal_states=None, annotation=None, trajectories=None,
                      hide_tick_labels=False, show_grid=False, grid_alpha=0.2):
    """Plot policy vector field with optional failure annotation and trajectory overlay"""
    setup_thesis_style()

    # grid_size is (H, W)
    H, W = int(grid_size[0]), int(grid_size[1])
    X, Y = np.meshgrid(np.arange(W), np.arange(H))  # X: columns (j), Y: rows (i)
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    sdim = int(H * W)

    for i in range(H):
        for j in range(W):
            idx = i * W + j
            state = torch.zeros(sdim).float()
            state[idx] = 1.0  # one-hot
            with torch.no_grad():
                try:
                    if policy is not None and hasattr(policy, '__call__'):
                        out = policy(state.unsqueeze(0))
                        # If the model returns a Distribution (e.g., torch.distributions.Categorical)
                        if hasattr(out, "probs"):
                            probs = out.probs  # (1, A) or (A,)
                            probs = probs[0] if probs.ndim == 2 else probs
                            action = int(torch.argmax(probs).item())
                        # If the model returns logits/tensor
                        elif torch.is_tensor(out):
                            logits = out[0] if out.ndim == 2 else out
                            probs = F.softmax(logits, dim=-1)
                            action = int(torch.argmax(probs).item())
                        else:
                            # Unknown output type — fall back gracefully
                            annotation = "Policy not callable — fallback arrows"
                            action = 0
                    else:
                        # Handle None policy case with fallback
                        annotation = "Policy not callable — fallback arrows"
                        action = 0  # default

                except Exception as e:
                    print(f"Warning: Error computing action for state {i},{j}: {e}")
                    action = 0  # default

            di, dj = ACTION_TO_DELTA.get(action, (0, 0))
            # Quiver expects Δx (columns) in U, Δy (rows) in V.
            U[i, j] = float(dj)
            # Y-axis is already inverted via set_ylim(H-0.5, -0.5); use di directly.
            V[i, j] = float(di)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    # Neutral board-style backdrop; explicit extents so tile centers are at integer coords
    extent = (-0.5, W - 0.5, H - 0.5, -0.5)
    ax.imshow(np.zeros((H, W)), origin='upper', extent=extent, interpolation='nearest', alpha=0.12)

    # Draw faint grid lines aligned to tile borders
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)  # origin='upper' without invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.arange(W))
    ax.set_yticks(np.arange(H))
    if hide_tick_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        ax.set_xticklabels([str(x) for x in range(1, W + 1)])
        ax.set_yticklabels([str(y) for y in range(1, H + 1)])
    # Optional board grid at cell borders (OFF by default to avoid stray lines)
    if show_grid:
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which='minor', linewidth=0.5, alpha=float(grid_alpha))
    ax.tick_params(which='major', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Quiver centered in each tile
    q = ax.quiver(X, Y, U, V, scale=1, scale_units='xy', angles='xy',
                  pivot='mid', width=0.006, zorder=2)

    if terminal_states:
        for (i, j) in terminal_states:
            ax.plot(j, i, 'r*', markersize=15, zorder=3)

    # Add failure annotation in top-left if policy loading failed
    if annotation:
        ax.text(0.02, 0.98, annotation, transform=ax.transAxes, fontsize=10,
                color='red', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Overlay trajectories if provided
    if trajectories is not None and len(trajectories) > 0:
        # Keep as list-of-episodes; do NOT coerce the whole structure to ndarray (can be ragged)
        def _step_to_ij(step):
            """Convert a step (scalar idx) or (i,j,...) tuple to (i,j), else None."""
            # Scalar index
            if isinstance(step, (int, np.integer)):
                idx = int(step)
                if 0 <= idx < H * W:
                    return divmod(idx, W)
                return None
            # Numpy scalar
            if isinstance(step, np.ndarray) and step.ndim == 0 and np.issubdtype(step.dtype, np.integer):
                idx = int(step.item())
                if 0 <= idx < H * W:
                    return divmod(idx, W)
                return None
            # Tuple/list/array with at least 2 elements
            if isinstance(step, (list, tuple, np.ndarray)) and len(step) >= 2:
                a, b = step[0], step[1]
                if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
                    return int(a), int(b)
            return None

        max_eps = 10
        drawn = 0
        for ep in trajectories:
            if drawn >= max_eps:
                break
            coords = []
            try:
                for step in ep:
                    ij = _step_to_ij(step)
                    if ij is not None:
                        i_s, j_s = ij
                        coords.append((j_s, i_s))  # (x, y)
            except Exception as e:
                print(f"Warning: skipping malformed trajectory: {e}")
                continue
            if len(coords) > 1:
                arr = np.array(coords, dtype=float)
                ax.plot(arr[:, 0], arr[:, 1], alpha=0.3, linewidth=1, color='magenta', zorder=1)
                drawn += 1

    ax.set_title("Learned Policy Field")
    save_figure(fig, save_path)
    print(f"[ok] Policy field plotted for grid {H}x{W} → {save_path}")

    plt.close('all')

def plot_policy_for_run(run_dir, out_dir, overlay_trajectories=False, hide_tick_labels=False, show_grid=False, grid_alpha=0.2):
    """Plot policy field for a single run"""
    env_data_path = os.path.join(run_dir, "env_data.json")
    if not os.path.exists(env_data_path):
        print(f"Skipping {run_dir}: env_data.json not found")
        # Attempt to include scenario in placeholder if possible
        try:
            cfg = load_config_with_fallback(run_dir)
            scenario_label = label_scenario(cfg) if cfg else os.path.basename(run_dir)
        except Exception:
            scenario_label = os.path.basename(run_dir)
        _save_placeholder(run_dir, out_dir, scenario_label, "env_data.json not found")
        return

    try:
        with open(env_data_path, 'r') as f:
            env_data = json.load(f)

        grid_size = env_data.get("grid_size") or env_data.get("grid_shape", [5, 5])
        # Accept both keys: 'terminal_states' (old) and 'terminals' (new)
        terminal_states = _normalize_terminals(
            env_data.get("terminal_states", []) or env_data.get("terminals", [])
        )

        # Scenario label for filenames and context
        try:
            cfg = load_config_with_fallback(run_dir)
            scenario_label = label_scenario(cfg) if cfg else os.path.basename(run_dir)
        except Exception:
            scenario_label = os.path.basename(run_dir)

        try:
            policy, annotation = load_policy(run_dir)
        except Exception as e:
            print(f"Policy load failed for {run_dir}: {e}")
            # Create fallback state for plotting
            policy = None
            annotation = "Policy not callable — fallback arrows"

        # Load trajectories if overlay requested
        trajectories = None
        if overlay_trajectories:
            traj_path = os.path.join(run_dir, 'trajectories.npy')
            if os.path.exists(traj_path):
                try:
                    trajectories = np.load(traj_path, allow_pickle=True)
                    print(f"Loaded {len(trajectories)} trajectories for overlay")
                except Exception as e:
                    print(f"Warning: Failed to load trajectories from {run_dir}: {e}")
            else:
                print(f"Warning: trajectories.npy not found in {run_dir}, skipping overlay")

        run_id = os.path.basename(run_dir.rstrip('/'))
        save_path = os.path.join(
            out_dir, f"{_sanitize(run_id)}__{_sanitize(scenario_label)}__policy_field.png"
        )

        plot_vector_field(policy, grid_size, save_path, terminal_states, annotation, trajectories, hide_tick_labels=hide_tick_labels, show_grid=show_grid, grid_alpha=grid_alpha)

    except Exception as e:
        print(f"Error processing {run_dir}: {e}")
        plt.close('all')

def main():
    parser = argparse.ArgumentParser(description="Plot learned policy as vector field")
    parser.add_argument('--roots', nargs='+', required=True, help='Root directories with runs')
    parser.add_argument('--out', required=True, help='Output directory under results/figures/policy')
    parser.add_argument('--overlay_trajectories', action='store_true', help='Overlay trajectories if trajectories.npy exists')
    # Show tick labels by default; user can hide them
    parser.add_argument('--hide_tick_labels', action='store_true', default=False, help='Hide axis tick labels for a cleaner board (default: off)')
    parser.add_argument('--show_tick_labels', dest='hide_tick_labels', action='store_false')
    # Optional board grid (off by default to avoid stray vertical line)
    parser.add_argument('--show_grid', action='store_true', default=False,
                        help='Draw a faint board grid at cell borders (default: off)')
    parser.add_argument('--grid_alpha', type=float, default=0.2,
                        help='Alpha for the optional board grid (default: 0.2)')

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    run_dirs = find_run_dirs(args.roots)
    if not run_dirs:
        print("No run directories found")
        return

    for run_dir in run_dirs:
        plot_policy_for_run(run_dir, args.out, args.overlay_trajectories, hide_tick_labels=args.hide_tick_labels, show_grid=args.show_grid, grid_alpha=args.grid_alpha)

if __name__ == '__main__':
    main()
