import numpy as np
from scipy.stats import pearsonr
from scipy.sparse import issparse, csr_matrix
from envs.environments import BaseEnv
from irl.maxent_irl import compute_policy_from_value


def reward_correlation(r_true: np.ndarray, r_learned: np.ndarray) -> float:
    """Pearson correlation between ground-truth and learned reward."""
    r_true_flat = r_true.flatten()
    r_learned_flat = r_learned.flatten()

    if r_true_flat is None or r_learned_flat is None:
        return -1.0  # Indicate missing data

    # Handle NaN/inf values
    if np.any(np.isnan(r_true_flat)) or np.any(np.isnan(r_learned_flat)):
        return 0.0

    # Handle constant arrays
    if np.all(r_true_flat == r_true_flat[0]) or np.all(r_learned_flat == r_learned_flat[0]):
        return 0.0
        
    return pearsonr(r_true_flat, r_learned_flat)[0]

def value_iteration(env, T, rewards, gamma=0.99, threshold=1e-6, max_iter=1000):
    n_states = T.shape[1]
    V = np.zeros(n_states)

    # Ensure rewards is flat 1D vector
    rewards_flat = rewards.flatten()
    if rewards_flat.shape != (n_states,):
        print(f"[WARN] rewards shape {rewards.shape} flattened to {rewards_flat.shape}")

    for _ in range(max_iter):
        V_prev = V.copy()
        if issparse(T):
            # Create and validate input vector
            input_vec = rewards_flat + gamma * V
            if input_vec.shape != (n_states,):
                raise ValueError(
                    f"Input vector must be shape ({n_states},). Got {input_vec.shape}"
                )
            if input_vec.ndim != 1:
                input_vec = input_vec.flatten()
                print(f"[WARN] Flattened input_vec to shape {input_vec.shape}")

            # Safe sparse multiplication
            Q_vals = T.dot(input_vec)
            if Q_vals.ndim > 1:
                Q_vals = Q_vals.A1  # Convert matrix to flat array

            # Validate output before reshape
            expected_size = env.n_states * env.n_actions
            if Q_vals.size != expected_size:
                raise ValueError(
                    f"Q_vals size mismatch: Expected {expected_size}, got {Q_vals.size}\n"
                    f"T.shape={T.shape}, input_vec.shape={input_vec.shape}"
                )

            Q = Q_vals.reshape(env.n_states, env.n_actions)  # Reshape using env dimensions
        else:
            Q_vals = np.array([
                T[:, a, :].dot(rewards + gamma * V)
                for a in range(env.n_actions)
            ]).T
            Q = Q_vals
        V = np.max(Q, axis=1)
        if np.max(np.abs(V - V_prev)) < threshold:
            break
    return V

def evaluate_irl_result(env, learned_reward, gamma, T=None):
    """
    Flexible evaluation for both discrete and continuous environments.
    Accepts precomputed transition matrix T to avoid rebuilding.
    """
    results = {}
    
    # Only compute these for GridWorld environments
    if hasattr(env, 'grid_size'):
        # Use precomputed T if available, else build
        if T is None:
            print("[Warning] Transition matrix T missing — rebuilding with default sparse=True.")
            print(f"[DEBUG] env type: {type(env)}")
            T = env.build_transition_matrix()
            assert issparse(T), "Expected sparse transition matrix for GridWorld evaluation"
        true_reward = env.get_ground_truth_reward().flatten()
        learned_reward_flat = learned_reward.flatten()
        
        # Compute true value function and policy
        V_true = value_iteration(env, T, true_reward, gamma)
        pi_true = compute_policy_from_value(T, true_reward, V_true, gamma)
        pi_true_actions = np.argmax(pi_true, axis=1)
        
        # Compute learned value function and policy
        V_learned = value_iteration(env, T, learned_reward_flat, gamma)
        pi_learned = compute_policy_from_value(T, learned_reward_flat, V_learned, gamma)
        pi_learned_actions = np.argmax(pi_learned, axis=1)
        
        results = {
            "reward_correlation": reward_correlation(true_reward, learned_reward_flat),
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
    if 'per_z_rewards' in additional_data and additional_data['per_z_rewards']:
        stacked = np.stack(additional_data['per_z_rewards'])
        return np.mean(np.var(stacked, axis=0))
    return 0  # Other methods don't have Z-dependent rewards
