"""
Environment utilities for Causal-AIRL Streamlit app.
====================================================
Helpers for creating and manipulating GridWorld environments.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any

# Import from main repo
try:
    from envs.environments import GridWorld, ConfoundedGridWorld, SlipperyGridWorld, build_env
    ENVS_AVAILABLE = True
except ImportError:
    ENVS_AVAILABLE = False


def create_env_from_params(
    grid_size: Tuple[int, int] = (5, 5),
    reward_type: str = "sparse",
    slip_prob: float = 0.0,
    gamma: float = 0.99,
    terminal_states: Optional[List[Tuple[int, int]]] = None,
    use_confounder: bool = False,
    confounder_value: Optional[int] = None,
    seed: Optional[int] = None
) -> Any:
    """
    Create a GridWorld environment from parameters.
    
    Args:
        grid_size: (rows, cols) tuple
        reward_type: "sparse" or "shaped"
        slip_prob: Stochastic transition probability
        gamma: Discount factor
        terminal_states: List of goal states; defaults to bottom-right corner
        use_confounder: Whether to create ConfoundedGridWorld
        confounder_value: Z value for confounded environment (0 or 1)
        seed: Random seed
        
    Returns:
        GridWorld or ConfoundedGridWorld instance
    """
    if not ENVS_AVAILABLE:
        raise ImportError("Environment modules not available. Run `pip install -e .` from repo root.")
    
    if terminal_states is None:
        terminal_states = [(grid_size[0] - 1, grid_size[1] - 1)]
    
    if use_confounder:
        return ConfoundedGridWorld(
            grid_size=grid_size,
            terminal_states=terminal_states,
            reward_type=reward_type,
            slip_prob=slip_prob,
            gamma=gamma,
            confounder_value=confounder_value,
            seed=seed
        )
    else:
        return GridWorld(
            grid_size=grid_size,
            terminal_states=terminal_states,
            reward_type=reward_type,
            reward_value=1.0,
            slip_prob=slip_prob,
            gamma=gamma,
            seed=seed
        )


def render_gridworld_state(
    env: Any,
    agent_pos: Optional[Tuple[int, int]] = None,
    show_rewards: bool = True
) -> np.ndarray:
    """
    Render GridWorld state as numpy array for visualisation.
    
    Args:
        env: GridWorld environment
        agent_pos: Current agent position (optional)
        show_rewards: Whether to show reward values
        
    Returns:
        2D numpy array representing the grid
    """
    if not hasattr(env, 'grid_size'):
        raise ValueError("Environment must have grid_size attribute")
    
    H, W = env.grid_size
    grid = np.zeros((H, W))
    
    if show_rewards:
        grid = env.get_ground_truth_reward()
    
    return grid


def get_env_info(env: Any) -> Dict[str, Any]:
    """
    Extract environment information for display.
    
    Args:
        env: GridWorld environment
        
    Returns:
        Dictionary with environment metadata
    """
    info = {
        'type': type(env).__name__,
        'grid_size': getattr(env, 'grid_size', (0, 0)),
        'n_states': getattr(env, 'n_states', 0),
        'n_actions': getattr(env, 'n_actions', 0),
        'reward_type': getattr(env, 'reward_type', 'unknown'),
        'slip_prob': getattr(env, 'slip_prob', 0.0),
        'gamma': getattr(env, 'gamma', 0.99),
        'terminal_states': list(getattr(env, 'terminal_states', [])),
    }
    
    if hasattr(env, 'confounder_values'):
        info['confounder_values'] = env.confounder_values
        info['current_z'] = getattr(env, 'z', None)
    
    return info


def generate_expert_demos(
    env: Any,
    n_trajectories: int = 20,
    optimality: str = "optimal",
    z: Optional[int] = None
) -> List[List[Tuple]]:
    """
    Generate expert demonstrations from environment.
    
    Args:
        env: GridWorld environment
        n_trajectories: Number of demonstration trajectories
        optimality: "optimal" or "random"
        z: Confounder value for ConfoundedGridWorld
        
    Returns:
        List of trajectories, each a list of (s, a, r, s') tuples
    """
    return env.sample_expert_trajectories(
        n_trajectories=n_trajectories,
        optimality=optimality,
        z=z
    )


def trajectory_stats(trajectories: List[List[Tuple]]) -> Dict[str, float]:
    """
    Compute statistics over a set of trajectories.
    
    Args:
        trajectories: List of trajectories
        
    Returns:
        Dictionary with trajectory statistics
    """
    if not trajectories:
        return {'count': 0}
    
    lengths = [len(t) for t in trajectories]
    total_rewards = [sum(step[2] for step in t) for t in trajectories]
    
    return {
        'count': len(trajectories),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
    }
