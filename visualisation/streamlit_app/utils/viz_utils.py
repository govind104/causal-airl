"""
Visualisation utilities for Causal-AIRL Streamlit app.
=======================================================
Plotting functions optimised for Streamlit display.
Includes vectorised policy visualisation for performance.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from typing import Optional, List, Tuple, Dict, Any

# Action deltas for policy visualisation
ACTION_DELTAS = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1),   # RIGHT
}


def plot_reward_heatmaps(
    true_reward: np.ndarray,
    learned_reward: np.ndarray,
    terminals: Optional[List[Tuple[int, int]]] = None,
    title_prefix: str = ""
) -> plt.Figure:
    """
    Create 3-panel reward heatmap: True | Learned | Difference.
    
    Args:
        true_reward: Ground truth reward array (H, W)
        learned_reward: Learned reward array (H, W)
        terminals: List of terminal state coordinates
        title_prefix: Optional prefix for titles
        
    Returns:
        matplotlib Figure for st.pyplot()
    """
    # Ensure same shape
    if true_reward.shape != learned_reward.shape:
        raise ValueError(f"Shape mismatch: true {true_reward.shape} vs learned {learned_reward.shape}")
    
    diff_reward = learned_reward - true_reward
    
    # Symmetric color scale
    max_abs = max(
        np.abs(true_reward).max(),
        np.abs(learned_reward).max(),
        np.abs(diff_reward).max()
    )
    if max_abs == 0:
        max_abs = 1.0
    vmin, vmax = -max_abs, max_abs
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Panel 1: True Reward
    im0 = axes[0].imshow(true_reward, cmap='RdBu_r', origin='upper', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'{title_prefix}True Reward')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Panel 2: Learned Reward
    im1 = axes[1].imshow(learned_reward, cmap='RdBu_r', origin='upper', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'{title_prefix}Learned Reward')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Panel 3: Difference
    im2 = axes[2].imshow(diff_reward, cmap='RdBu_r', origin='upper', vmin=vmin, vmax=vmax)
    axes[2].set_title(f'{title_prefix}Difference (Learned - True)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    # Overlay terminal states
    if terminals:
        for (i, j) in terminals:
            for ax in axes:
                ax.plot(j, i, marker='*', markersize=12, 
                       markeredgecolor='white', color='black')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_policy_with_trajectories(
    policy: Any,
    grid_size: Tuple[int, int],
    trajectories: Optional[List] = None,
    terminals: Optional[List[Tuple[int, int]]] = None,
    title: str = "Policy Vector Field"
) -> plt.Figure:
    """
    Plot learned policy as vector field with optional trajectory overlay.
    Uses VECTORISED inference for performance on large grids.
    
    Args:
        policy: PolicyNet or callable that returns action distribution
        grid_size: (H, W) tuple
        trajectories: Optional list of trajectories to overlay
        terminals: List of terminal state coordinates
        title: Plot title
        
    Returns:
        matplotlib Figure for st.pyplot()
    """
    H, W = grid_size
    n_states = H * W
    
    # Initialize grids
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    
    # VECTORISED: Create all one-hot states at once
    if policy is not None and hasattr(policy, '__call__'):
        try:
            # Batch inference: all states at once
            states = torch.eye(n_states, dtype=torch.float32)
            
            with torch.no_grad():
                out = policy(states)
                
                # Handle different output types
                if hasattr(out, 'probs'):
                    probs = out.probs  # (n_states, n_actions)
                    actions = probs.argmax(dim=1).numpy()
                elif torch.is_tensor(out):
                    actions = out.argmax(dim=1).numpy()
                else:
                    actions = np.zeros(n_states, dtype=int)
            
            # Map actions to U, V grids
            for idx in range(n_states):
                i, j = divmod(idx, W)
                di, dj = ACTION_DELTAS.get(int(actions[idx]), (0, 0))
                U[i, j] = float(dj)
                V[i, j] = float(di)
                
        except Exception as e:
            print(f"Vectorised inference failed: {e}, falling back to loop")
            # Fallback to per-state inference
            for i in range(H):
                for j in range(W):
                    idx = i * W + j
                    state = torch.zeros(n_states)
                    state[idx] = 1.0
                    
                    try:
                        with torch.no_grad():
                            out = policy(state.unsqueeze(0))
                            if hasattr(out, 'probs'):
                                action = out.probs[0].argmax().item()
                            else:
                                action = out[0].argmax().item()
                    except Exception:
                        action = 0
                    
                    di, dj = ACTION_DELTAS.get(action, (0, 0))
                    U[i, j] = float(dj)
                    V[i, j] = float(di)
    else:
        # No policy - show random arrows for demo
        for i in range(H):
            for j in range(W):
                action = np.random.randint(4)
                di, dj = ACTION_DELTAS.get(action, (0, 0))
                U[i, j] = float(dj)
                V[i, j] = float(di)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Background grid
    ax.imshow(np.zeros((H, W)), origin='upper', alpha=0.1, cmap='gray',
              extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    
    # Policy arrows (quiver)
    ax.quiver(X, Y, U, V, scale=1, scale_units='xy', angles='xy', 
              pivot='mid', width=0.015, color='steelblue', zorder=2)
    
    # Overlay trajectories
    if trajectories is not None and len(trajectories) > 0:
        max_traj = min(10, len(trajectories))
        for traj in trajectories[:max_traj]:
            coords = []
            for step in traj:
                # Handle different trajectory formats
                if isinstance(step, (list, tuple)) and len(step) >= 1:
                    s = step[0]
                    if isinstance(s, (list, tuple, np.ndarray)) and len(s) >= 2:
                        coords.append((float(s[1]), float(s[0])))  # (x, y)
                    elif isinstance(s, (int, np.integer)):
                        i, j = divmod(int(s), W)
                        coords.append((float(j), float(i)))
            
            if len(coords) > 1:
                arr = np.array(coords)
                ax.plot(arr[:, 0], arr[:, 1], alpha=0.5, linewidth=2, 
                       color='magenta', zorder=1)
    
    # Terminal states
    if terminals:
        for (i, j) in terminals:
            ax.plot(j, i, 'r*', markersize=20, zorder=10)
    
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_curves(
    training_logs: Dict[str, Any],
    metrics: Optional[List[str]] = None,
    smooth_window: int = 5
) -> plt.Figure:
    """
    Plot training curves from training logs.
    
    Args:
        training_logs: Dictionary with metric time series
        metrics: List of metrics to plot (auto-detect if None)
        smooth_window: Moving average window for smoothing
        
    Returns:
        matplotlib Figure for st.pyplot()
    """
    # Auto-detect metrics if not specified
    if metrics is None:
        # Common metric names
        candidates = [
            'discriminator_loss', 'disc_bce', 'D_loss',
            'policy_loss', 'pi_loss',
            'epoch_inv_loss', 'invariance_loss',
            'epoch_kl_raw', 'epoch_kl_post',
            'reward_correlation', 'policy_agreement'
        ]
        metrics = [m for m in candidates if m in training_logs and 
                   isinstance(training_logs[m], (list, np.ndarray)) and 
                   len(training_logs[m]) > 1]
    
    if not metrics:
        # Create placeholder
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No training metrics available", ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        return fig
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    def moving_average(arr, window):
        if window <= 1 or len(arr) < window:
            return arr
        return np.convolve(arr, np.ones(window)/window, mode='valid')
    
    for ax, metric in zip(axes, metrics):
        values = np.array(training_logs[metric])
        
        # Filter invalid values
        valid_mask = np.isfinite(values)
        if not np.any(valid_mask):
            ax.text(0.5, 0.5, f"{metric}\n(no valid data)", ha='center', va='center',
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(metric)
            continue
        
        values = values[valid_mask]
        
        # Plot raw
        ax.plot(values, alpha=0.3, color='gray', linewidth=0.5)
        
        # Plot smoothed
        if len(values) >= smooth_window:
            smoothed = moving_average(values, smooth_window)
            ax.plot(np.arange(len(smoothed)) + smooth_window // 2, smoothed, 
                   color='steelblue', linewidth=1.5, label=f'{metric} (MA={smooth_window})')
        else:
            ax.plot(values, color='steelblue', linewidth=1.5, label=metric)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    
    plt.tight_layout()
    return fig


def plot_single_reward(
    reward: np.ndarray,
    terminals: Optional[List[Tuple[int, int]]] = None,
    title: str = "Reward",
    cmap: str = 'RdBu_r'
) -> plt.Figure:
    """
    Plot a single reward heatmap.
    
    Args:
        reward: 2D reward array
        terminals: Terminal state coordinates
        title: Plot title
        cmap: Colormap name
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    max_abs = np.abs(reward).max()
    if max_abs == 0:
        max_abs = 1.0
    
    im = ax.imshow(reward, cmap=cmap, origin='upper', vmin=-max_abs, vmax=max_abs)
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    if terminals:
        for (i, j) in terminals:
            ax.plot(j, i, 'r*', markersize=15)
    
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    return fig
