import os
import numpy as np
import torch

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

def evaluate_continuous_env(env, learned_reward, policy, cfg):
    """Compute meaningful metrics for continuous environments like CartPole"""
    metrics = {}

    # 1. Collect episode data AND sample states for reward correlation
    episode_returns = []
    episode_lengths = []
    expert_actions = []
    learned_actions = []
    state_samples = []

    for _ in range(10):  # Test over multiple episodes
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        total_return = 0
        steps = 0
        done = False

        while not done and steps < 500:
            # Collect state for reward correlation
            state_samples.append(obs.copy())

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_dist = policy(obs_tensor)
                action = action_dist.sample().item()
            learned_actions.append(action)

            # Get expert action for policy agreement
            expert_action = env.expert_policy(obs, mode='optimal') if hasattr(env, 'expert_policy') else action
            expert_actions.append(expert_action)

            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_return += 1 if not done else 0  # Basic survival reward
            obs = next_obs
            steps += 1

        episode_returns.append(total_return)
        episode_lengths.append(steps)

    metrics.update({
        "reward_correlation": None,  # Not applicable for continuous
        "value_difference": None,    # Not applicable for continuous
        "policy_agreement": None,    # Not applicable for continuous
        "continuous": True,
        "avg_episode_return": np.mean(episode_returns),
        "avg_episode_length": np.mean(episode_lengths),
        "episode_return_std": np.std(episode_returns)
    })

    return metrics

def evaluate_irl_result(env, cfg, save_dir, learned_reward, gamma, T=None, heldout_mask=None):
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

        # Optional held-out region masking (train/test split)
        train_mask = nonterminal_mask.copy()
        test_mask = None
        if heldout_mask is not None:
            heldout_mask = np.asarray(heldout_mask, dtype=bool).reshape(-1)
            assert heldout_mask.shape[0] == env.n_states, "heldout_mask shape mismatch"
            test_mask = heldout_mask & nonterminal_mask
            train_mask = (~heldout_mask) & nonterminal_mask

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

        # Save value function if requested
        if cfg.get('eval', {}).get('save_value_function', False):
            # Save 1D value function (preserving contract)
            np.save(os.path.join(save_dir, 'V.npy'), V_learned)

            # Save 2D version for direct plotting if GridWorld
            if hasattr(env, 'grid_size'):
                H, W = env.grid_size
                if len(V_learned) == H * W:
                    np.save(os.path.join(save_dir, 'V_map.npy'), V_learned.reshape(H, W))

        results = {
            "reward_correlation": reward_correlation(true_reward, learned_reward_flat, mask=nonterminal_mask),
            "reward_corr_train": reward_correlation(true_reward, learned_reward_flat, mask=train_mask),
            "reward_corr_test": reward_correlation(true_reward, learned_reward_flat, mask=test_mask) if test_mask is not None else np.nan,
            "value_difference": np.mean(np.abs(V_true - V_learned)),
            "policy_agreement": (pi_true_actions == pi_learned_actions).mean(),
            "continuous": False
        }

    # Check if this is a continuous environment (like CartPole)
    elif hasattr(env, 'observation_space') and len(env.observation_space.shape) > 0:
        # This is likely a continuous environment
        # Load the learned policy if available
        policy_path = os.path.join(save_dir, 'policy.pt')
        model_weights_path = os.path.join(save_dir, 'model_weights.pt')

        if os.path.exists(policy_path):
            # Reconstruct the policy architecture
            from models.policy import PolicyNet
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            policy = PolicyNet(state_dim, action_dim)
            policy.load_state_dict(torch.load(policy_path, map_location='cpu'))
            policy.eval()

            return evaluate_continuous_env(env, learned_reward, policy, cfg)

        # ADD THIS BLOCK FOR CARTPOLE MODEL_WEIGHTS.PT
        elif os.path.exists(model_weights_path):
            # Load from model_weights.pt (CartPole case)
            from models.policy import PolicyNet
            model_data = torch.load(model_weights_path, map_location='cpu')

            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            policy = PolicyNet(state_dim, action_dim)
            policy.load_state_dict(model_data['policy'])
            policy.eval()

            return evaluate_continuous_env(env, learned_reward, policy, cfg)

        else:
            return {
                "reward_correlation": None,
                "value_difference": None,
                "policy_agreement": None,
                "continuous": True,
                "avg_episode_return": None,
                "avg_episode_length": None,
                "episode_return_std": None
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
