import numpy as np
from scipy.stats import pearsonr
from scipy.sparse import issparse
from irl.maxent_irl import compute_policy_from_value


def reward_correlation(r_true: np.ndarray, r_learned: np.ndarray, mask: np.ndarray) -> float:
    """Pearson correlation between ground-truth and learned state-only reward.

    Args:
        r_true: Ground truth reward vector
        r_learned: Learned state-only reward vector
        mask: Boolean mask for non-terminal states

    Returns:
        Pearson correlation coefficient, or np.nan if computation fails
    """
    # Require mask, do not fall back to reward-based heuristics
    if mask is None:
        raise ValueError("Non-terminal mask is required for reward correlation computation.")

    r_true_masked = r_true[mask]
    r_learned_masked = r_learned[mask]

    # Sparse-reward safe fallback: if the masked true rewards are constant (e.g., all zeros),
    # compute correlation over ALL states to include the terminal "1" signal.
    if np.std(r_true_masked) == 0:
        print("[INFO] Zero variance in masked true rewards — using ALL states for correlation")
        r_true_masked = r_true
        r_learned_masked = r_learned

    # Handle degenerate cases properly (after fallback)
    if len(r_true_masked) == 0 or len(r_learned_masked) == 0:
        print("[WARNING] Empty masked reward arrays - cannot compute correlation")
        return np.nan

    if np.any(np.isnan(r_true_masked)) or np.any(np.isnan(r_learned_masked)):
        print("[WARNING] NaN detected in reward arrays - cannot compute correlation")
        return np.nan

    if np.std(r_true_masked) == 0 or np.std(r_learned_masked) == 0:
        print("[WARNING] Zero variance in reward arrays - cannot compute correlation")
        return np.nan

    return pearsonr(r_true_masked, r_learned_masked)[0]

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
            T = env.build_transition_matrix()
            assert issparse(T), "Expected sparse transition matrix for GridWorld evaluation"
        true_reward = env.get_ground_truth_reward_vector()
        if not hasattr(env, 'get_nonterminal_mask'):
            raise AttributeError("Environment must implement get_nonterminal_mask() for evaluation")
        nonterminal_mask = env.get_nonterminal_mask()
        learned_reward_flat = np.asarray(learned_reward, dtype=float).reshape(-1)
        true_reward = np.asarray(true_reward, dtype=float).reshape(-1)

        # n_states consistency
        assert len(true_reward) == len(nonterminal_mask) == env.n_states
        # Transition matrix shape
        assert T.shape == (env.n_states * env.n_actions, env.n_states)

        # Sanity check lengths
        if len(true_reward) != len(learned_reward_flat) or len(nonterminal_mask) != len(true_reward):
            raise ValueError(f"Length mismatch in reward correlation inputs:\n"
                             f"true_reward length {len(true_reward)}, learned_reward length {len(learned_reward_flat)}, mask length {len(nonterminal_mask)}")
        
        # Compute true value function and policy
        V_true = value_iteration(env, T, true_reward, gamma)
        pi_true = compute_policy_from_value(T, true_reward, V_true, gamma)
        pi_true_actions = np.argmax(pi_true, axis=1)
        
        # Compute learned value function and policy
        V_learned = value_iteration(env, T, learned_reward_flat, gamma)
        pi_learned = compute_policy_from_value(T, learned_reward_flat, V_learned, gamma)
        pi_learned_actions = np.argmax(pi_learned, axis=1)
        
        results = {
            "reward_correlation": reward_correlation(true_reward, learned_reward_flat, mask=nonterminal_mask),
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
            expert_states.add(tuple(s) if hasattr(s, '__iter__') else (s,))
    
    learned_states = set()
    for traj in learned_trajs:
        for (s, a) in traj:
            learned_states.add(tuple(s) if hasattr(s, '__iter__') else (s,))
    
    intersection = expert_states & learned_states
    union = expert_states | learned_states
    return len(intersection) / len(union) if union else 0
