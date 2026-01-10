"""
Metrics utilities for Causal-AIRL Streamlit app.
=================================================
Evaluation metrics computation and aggregation.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from typing import Dict, Any, List, Optional


def get_metric_safe(data: dict, key: str, default=None) -> Any:
    """
    Safely get a metric from a dict with fallback to alternative keys.
    Handles list values by extracting first element.
    
    Args:
        data: Dictionary to search
        key: Primary key to look for
        default: Default value if not found
        
    Returns:
        Metric value or default
    """
    if data is None:
        return default
    
    def _extract_scalar(val):
        """Extract scalar from value that might be a list."""
        if val is None:
            return None
        if isinstance(val, (list, tuple)) and len(val) > 0:
            return _extract_scalar(val[0])  # Recursively extract first element
        if isinstance(val, np.ndarray):
            if val.size == 1:
                return float(val.flat[0])
            elif val.size > 1:
                return float(val.flat[0])
        return val
    
    # Try direct access first
    if key in data:
        val = _extract_scalar(data[key])
        if val is not None:
            return val
    
    # Common alternative keys mapping
    alt_keys = {
        'policy_agreement': ['final_policy_agreement', 'agreement', 'pi_agreement', 
                             'policy_agree', 'pi_agree', 'policy_agreement_all'],
        'reward_spearman': ['final_reward_spearman', 'spearman', 'rho', 'reward_rho',
                            'spearman_correlation', 'reward_rank_corr'],
        'reward_correlation': ['final_reward_correlation', 'pearson', 'reward_pearson', 
                               'r', 'correlation', 'reward_corr'],
        'value_correlation': ['final_value_correlation', 'value_corr', 'v_corr',
                              'value_agreement', 'v_agreement'],
        'wall_time': ['final_wall_time_sec', 'training_time', 'elapsed_time',
                      'total_time', 'time_sec'],
    }
    
    for alt in alt_keys.get(key, []):
        if alt in data:
            val = _extract_scalar(data[alt])
            if val is not None:
                return val
    
    return default


def compute_spearman_from_rewards(
    true_rewards: np.ndarray, 
    learned_rewards: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute Spearman rank correlation from reward arrays.
    
    Args:
        true_rewards: Ground truth rewards (flattened or 2D)
        learned_rewards: Learned rewards (flattened or 2D)
        mask: Optional boolean mask (e.g., exclude terminals)
        
    Returns:
        Spearman correlation coefficient
    """
    true_flat = true_rewards.flatten()
    learned_flat = learned_rewards.flatten()
    
    if mask is not None:
        mask_flat = mask.flatten()
        true_flat = true_flat[mask_flat]
        learned_flat = learned_flat[mask_flat]
    
    if len(true_flat) == 0 or len(learned_flat) == 0:
        return np.nan
    
    if np.std(true_flat) == 0 or np.std(learned_flat) == 0:
        return np.nan
    
    return spearmanr(true_flat, learned_flat).correlation


def compute_pearson_from_rewards(
    true_rewards: np.ndarray,
    learned_rewards: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute Pearson correlation from reward arrays.
    
    Args:
        true_rewards: Ground truth rewards
        learned_rewards: Learned rewards
        mask: Optional boolean mask
        
    Returns:
        Pearson correlation coefficient
    """
    true_flat = true_rewards.flatten()
    learned_flat = learned_rewards.flatten()
    
    if mask is not None:
        mask_flat = mask.flatten()
        true_flat = true_flat[mask_flat]
        learned_flat = learned_flat[mask_flat]
    
    if len(true_flat) == 0 or len(learned_flat) == 0:
        return np.nan
    
    if np.std(true_flat) == 0 or np.std(learned_flat) == 0:
        return np.nan
    
    return pearsonr(true_flat, learned_flat)[0]


def compute_spearman_from_trajectories(
    true_rewards: np.ndarray,
    trajectories: List[List],
    grid_size: tuple
) -> float:
    """
    Compute Spearman correlation from trajectory visitation.
    
    Args:
        true_rewards: Ground truth rewards (H, W)
        trajectories: List of trajectories
        grid_size: (H, W) tuple
        
    Returns:
        Spearman correlation coefficient
    """
    H, W = grid_size
    
    # Compute visitation counts
    visitation = np.zeros((H, W))
    for traj in trajectories:
        for step in traj:
            if isinstance(step, (list, tuple)) and len(step) >= 1:
                s = step[0]
                if isinstance(s, (list, tuple, np.ndarray)) and len(s) >= 2:
                    i, j = int(s[0]), int(s[1])
                    if 0 <= i < H and 0 <= j < W:
                        visitation[i, j] += 1
    
    # Normalize
    if visitation.sum() > 0:
        visitation = visitation / visitation.sum()
    
    return compute_spearman_from_rewards(true_rewards, visitation)


def load_metrics_from_run(run_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load metrics.json from a run directory.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Dictionary with metrics, or None if not found
    """
    metrics_path = os.path.join(run_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        return None
    
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Add run metadata
        metrics['run_path'] = run_dir
        metrics['run_name'] = os.path.basename(run_dir)
        
        # Extract method and scenario from config if available
        config_path = os.path.join(run_dir, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                metrics['method'] = config.get('irl', {}).get('method', 'unknown')
                metrics['scenario'] = _infer_scenario(config, run_dir)
            except Exception:
                pass
        
        return metrics
    except Exception:
        return None


def _infer_scenario(config: Dict, run_dir: str) -> str:
    """Infer scenario label from config and path."""
    parts = []
    
    env_cfg = config.get('env', {})
    if env_cfg.get('name', '') == 'ConfoundedGridWorld':
        parts.append('confounded')
        z = env_cfg.get('confounder_value')
        if z is not None:
            parts.append(f'z{z}')
    
    slip = env_cfg.get('slip_prob', 0.0)
    if slip > 0:
        parts.append(f'slip{slip}')
    
    reward_type = env_cfg.get('reward_type', '')
    if reward_type and reward_type != 'sparse':
        parts.append(reward_type)
    
    if not parts:
        # Use directory name
        return os.path.basename(run_dir)
    
    return '_'.join(parts)


def load_all_metrics(run_dirs: List[str]) -> List[Dict[str, Any]]:
    """
    Load metrics from all run directories.
    
    Args:
        run_dirs: List of paths to run directories
        
    Returns:
        List of metrics dictionaries
    """
    all_metrics = []
    for run_dir in run_dirs:
        metrics = load_metrics_from_run(run_dir)
        if metrics is not None:
            all_metrics.append(metrics)
    return all_metrics


def compute_summary_table(all_metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build summary DataFrame from all metrics.
    
    Args:
        all_metrics: List of metrics dictionaries
        
    Returns:
        DataFrame with summary statistics
    """
    rows = []
    
    for m in all_metrics:
        row = {
            'Run': m.get('run_name', 'unknown'),
            'Method': m.get('method', 'unknown'),
            'Scenario': m.get('scenario', ''),
            'Reward Spearman': m.get('reward_spearman', m.get('final_reward_spearman', np.nan)),
            'Reward Pearson': m.get('reward_correlation', m.get('final_reward_correlation', np.nan)),
            'Policy Agreement': m.get('policy_agreement', m.get('final_policy_agreement', np.nan)),
            'Value Correlation': m.get('value_correlation', np.nan),
            'Wall Time (s)': m.get('final_wall_time_sec', m.get('wall_time', np.nan)),
        }
        
        # Cross-Z metrics if available
        for key in m.keys():
            if 'cross_z' in key.lower():
                row[key] = m[key]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by method, then scenario
    df = df.sort_values(['Method', 'Scenario']).reset_index(drop=True)
    
    return df


def plot_crossz_comparison(
    airl_metrics: Dict[str, Any],
    cairl_metrics: Dict[str, Any]
) -> plt.Figure:
    """
    Bar chart comparing cross-Z generalization between methods.
    
    Args:
        airl_metrics: Metrics dict for AIRL run
        cairl_metrics: Metrics dict for Causal-AIRL run
        
    Returns:
        matplotlib Figure
    """
    # Extract cross-Z values
    def get_crossz(m, from_z, to_z):
        keys = [
            f'cross_z_from_{from_z}_to_{to_z}',
            f'cross_z_{from_z}_to_{to_z}',
            f'crossz_{from_z}_{to_z}'
        ]
        for k in keys:
            if k in m:
                return float(m[k])
        return 0.0
    
    labels = ['Z=0→Z=1', 'Z=1→Z=0']
    
    airl_vals = [
        get_crossz(airl_metrics, 0, 1),
        get_crossz(airl_metrics, 1, 0)
    ]
    cairl_vals = [
        get_crossz(cairl_metrics, 0, 1),
        get_crossz(cairl_metrics, 1, 0)
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars1 = ax.bar(x - width/2, airl_vals, width, label='AIRL', color='coral')
    bars2 = ax.bar(x + width/2, cairl_vals, width, label='Causal-AIRL', color='steelblue')
    
    ax.set_ylabel('Policy Agreement')
    ax.set_xlabel('Train Z → Test Z')
    ax.set_title('Cross-Z Generalization Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def compute_method_comparison(
    all_metrics: List[Dict[str, Any]],
    metric_name: str = 'reward_spearman'
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and std of a metric grouped by method.
    
    Args:
        all_metrics: List of metrics dictionaries
        metric_name: Name of metric to compare
        
    Returns:
        Dictionary mapping method to {mean, std, count}
    """
    from collections import defaultdict
    
    by_method = defaultdict(list)
    
    for m in all_metrics:
        method = m.get('method', 'unknown')
        value = m.get(metric_name, m.get(f'final_{metric_name}'))
        if value is not None and not np.isnan(value):
            by_method[method].append(float(value))
    
    result = {}
    for method, values in by_method.items():
        result[method] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'count': len(values)
        }
    
    return result
