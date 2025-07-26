import numpy as np
from scipy.stats import pearsonr
from envs.utils import build_transition_matrix
from irl.maxent_irl import compute_policy_from_value, soft_value_iteration as value_iteration

def reward_correlation(r_true: np.ndarray, r_learned: np.ndarray) -> float:
    """Pearson correlation between ground-truth and learned reward."""
    if r_true is None or r_learned is None:
        return -1.0  # Indicate missing data
    
    r_true_flat = r_true.flatten()
    r_learned_flat = r_learned.flatten()
    
    # Handle constant arrays
    if np.all(r_true_flat == r_true_flat[0]) or np.all(r_learned_flat == r_learned_flat[0]):
        return 0.0
        
    return pearsonr(r_true_flat, r_learned_flat)[0]

def evaluate_irl_result(env, learned_reward, expert_policy, gamma):
    """
    Flexible evaluation for both discrete and continuous environments.
    Returns placeholder metrics for continuous environments.
    """
    results = {}
    
    # Only compute these for gridworld environments
    if hasattr(env, 'grid_size'):
        T = build_transition_matrix(env.grid_size, slip_prob=env.slip_prob)
        true_reward = env.get_ground_truth_reward().flatten()
        n_states = env.n_states
        
        # Compute true value function and policy
        V_true = value_iteration(T, true_reward, gamma)
        pi_true = compute_policy_from_value(T, true_reward, V_true, gamma)
        pi_true_actions = np.argmax(pi_true, axis=1)
        
        # Compute learned value function and policy
        V_learned = value_iteration(T, learned_reward, gamma)
        pi_learned = compute_policy_from_value(T, learned_reward, V_learned, gamma)
        pi_learned_actions = np.argmax(pi_learned, axis=1)
        
        results = {
            "reward_correlation": reward_correlation(true_reward, learned_reward),
            "value_difference": np.mean(np.abs(V_true - V_learned)),
            "policy_agreement": (pi_true_actions == pi_learned_actions).mean(),
            "continuous": False
        }
    else:
        # Placeholder metrics for continuous environments
        results = {
            "reward_correlation": -1.0,
            "value_difference": -1.0,
            "policy_agreement": -1.0,
            "continuous": True
        }
    
    return results

def compute_trajectory_overlap(expert_trajs, learned_trajs):
    """Compute Jaccard similarity of visited states"""
    expert_states = set()
    for traj in expert_trajs:
        for (s, a, r, s_next) in traj:
            expert_states.add(tuple(s))
    
    learned_states = set()
    for traj in learned_trajs:
        for (s, a) in traj:
            learned_states.add(tuple(s))
    
    intersection = expert_states & learned_states
    union = expert_states | learned_states
    return len(intersection) / len(union) if union else 0

def compute_reward_variance(additional_data):
    """Compute reward variance across Z values"""
    if 'invariant_reward' in additional_data and 'causal_reward' in additional_data:
        # For Causal-AIRL
        return np.var(additional_data['causal_reward'])
    return 0  # Other methods don't have Z-dependent rewards