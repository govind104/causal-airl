import os
import numpy as np
import torch
import glob

from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse
from irl.maxent_irl import compute_policy_from_value
from models.policy import PolicyNet


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

def reward_rank_correlation(r_true: np.ndarray, r_learned: np.ndarray, mask: np.ndarray) -> float:
    """Spearman rank correlation with identical masking semantics as reward_correlation."""
    if mask is None:
        raise ValueError("Non-terminal mask is required for reward rank correlation.")
    r_true_masked = r_true[mask]
    r_learned_masked = r_learned[mask]
    if np.std(r_true_masked) == 0:
        r_true_masked = r_true
        r_learned_masked = r_learned
    if len(r_true_masked) == 0 or len(r_learned_masked) == 0:
        return np.nan
    if np.any(np.isnan(r_true_masked)) or np.any(np.isnan(r_learned_masked)):
        return np.nan
    if np.std(r_true_masked) == 0 or np.std(r_learned_masked) == 0:
        return np.nan
    return spearmanr(r_true_masked, r_learned_masked).correlation

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

def _rollout_and_score(env, policy, z=None, episodes=10):
    """Helper to run episodes and score performance"""
    returns, lengths = [], []
    expert_agreements = []

    for _ in range(episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        if z is not None and hasattr(env, 'set_confounder'):
            env.set_confounder(z)

        total_return = 0
        steps = 0
        done = False
        agreements_this_ep = []

        while not done and steps < 500:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_dist = policy(obs_tensor)
                action = action_dist.sample().item()

            # Expert agreement (respect z if supported; be robust to different kw names)
            if hasattr(env, 'expert_policy'):
                expert_action = None
                if z is not None:
                    # Try (z, optimality), then (z, mode), then fallbacks
                    try:
                        expert_action = env.expert_policy(obs, z=z, optimality='optimal')
                    except TypeError:
                        try:
                            expert_action = env.expert_policy(obs, z=z, mode='optimal')
                        except TypeError:
                            pass
                if expert_action is None:
                    # No z, or z not accepted — try without z
                    try:
                        expert_action = env.expert_policy(obs, optimality='optimal')
                    except TypeError:
                        expert_action = env.expert_policy(obs, mode='optimal')
                agreements_this_ep.append(action == expert_action)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_return += float(reward)
            obs = next_obs
            steps += 1

        returns.append(total_return)
        lengths.append(steps)
        if agreements_this_ep:
            expert_agreements.append(np.mean(agreements_this_ep))

    return returns, lengths, np.mean(expert_agreements) if expert_agreements else np.nan

def evaluate_continuous_env(env, learned_reward, policy, cfg):
    """Compute meaningful metrics for continuous environments like CartPole"""
    metrics = {}

    # Check if environment has confounders for per-z evaluation
    if hasattr(env, 'confounder_values') and env.confounder_values:
        per_z_metrics = {}

        for z in env.confounder_values:
            z_returns, z_lengths, z_agreement = _rollout_and_score(env, policy, z, episodes=5)
            per_z_metrics[f"avg_return_z{z}"] = np.mean(z_returns)
            per_z_metrics[f"avg_length_z{z}"] = np.mean(z_lengths)
            per_z_metrics[f"expert_agreement_z{z}"] = z_agreement

        metrics.update(per_z_metrics)

        # Aggregate metrics
        all_returns = [per_z_metrics[k] for k in per_z_metrics if k.startswith("avg_return_z")]
        metrics["avg_episode_return"] = np.mean(all_returns)
        metrics["return_variance_across_z"] = np.var(all_returns)
    else:
        # Original single-condition evaluation
        returns, lengths, agreement = _rollout_and_score(env, policy, z=None, episodes=10)
        metrics.update({
            "avg_episode_return": np.mean(returns),
            "avg_episode_length": np.mean(lengths),
            "episode_return_std": np.std(returns)
        })

    metrics.update({
        "reward_correlation": None,
        "value_difference": None,
        "policy_agreement": None,
        "continuous": True
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

        # Compute confounded expert agreement if applicable
        confounded_expert_agreement = np.nan
        confounded_expert_agreement_train = np.nan
        confounded_expert_agreement_test = np.nan

        if hasattr(env, 'expert_policy') and hasattr(env, 'get_current_confounder'):
            current_z = env.get_current_confounder()
            if current_z is not None:
                # Get confounded expert policy for all states
                confounded_expert_actions = np.zeros(env.n_states, dtype=int)
                for i in range(env.n_states):
                    state_coords = (i // env.grid_size[1], i % env.grid_size[1])
                    confounded_expert_actions[i] = env.expert_policy(state_coords, z=current_z)

                confounded_expert_agreement = (pi_learned_actions == confounded_expert_actions).mean()
                confounded_expert_agreement_train = (pi_learned_actions[train_mask] == confounded_expert_actions[train_mask]).mean() if train_mask is not None else np.nan
                confounded_expert_agreement_test = (pi_learned_actions[test_mask] == confounded_expert_actions[test_mask]).mean() if test_mask is not None else np.nan

        # Then add to results dict:
        results.update({
            "confounded_expert_agreement": confounded_expert_agreement,
            "confounded_expert_agreement_train": confounded_expert_agreement_train,
            "confounded_expert_agreement_test": confounded_expert_agreement_test,
        })

        # Cross-Z generalization metrics
        if hasattr(env, 'confounder_values') and len(env.confounder_values) > 1:
            cross_z_agreements = {}
            train_z = cfg.get('expert', {}).get('confounder_value', None)
            cfg_test = cfg.get('eval', {}).get('test_z', None)
            test_zs = [cfg_test] if (cfg_test is not None) else [z for z in env.confounder_values if z != train_z]
            for test_z in test_zs:
                # Get expert policy for test_z
                expert_actions_test = np.zeros(env.n_states, dtype=int)
                for i in range(env.n_states):
                    state_coords = (i // env.grid_size[1], i % env.grid_size[1])
                    expert_actions_test[i] = env.expert_policy(state_coords, z=test_z)

                agreement = float((pi_learned_actions == expert_actions_test).mean())
                key = f"cross_z_from_{train_z}_to_{test_z}" if train_z is not None else f"cross_z_to_{test_z}"
                cross_z_agreements[key] = agreement

            results.update(cross_z_agreements)

        # Sample-efficiency proxy via checkpoints
        ckpt_dir = os.path.join(save_dir, 'checkpoints')
        if os.path.isdir(ckpt_dir):
            for f in sorted(glob.glob(os.path.join(ckpt_dir, 'reward_iter_*.npy'))):
                try:
                    r_ckpt = np.load(f)
                    V_ckpt = value_iteration(env, T, r_ckpt, gamma)
                    pi_ckpt = compute_policy_from_value(T, r_ckpt, V_ckpt, gamma)
                    pi_ckpt_actions = np.argmax(pi_ckpt, axis=1)
                    pa_ckpt = float((pi_true_actions == pi_ckpt_actions).mean())
                    fname = os.path.basename(f).replace('.npy', '')
                    results[f"checkpoint_policy_agreement/{fname}"] = pa_ckpt
                except Exception as e:
                    print(f"[WARN] Checkpoint eval failed for {f}: {e}")

        # Save value function if requested
        if cfg.get('eval', {}).get('save_value_function', False):
            # Save 1D value function (preserving contract)
            np.save(os.path.join(save_dir, 'V.npy'), V_learned)

            # Save 2D version for direct plotting if GridWorld
            if hasattr(env, 'grid_size'):
                H, W = env.grid_size
                if len(V_learned) == H * W:
                    np.save(os.path.join(save_dir, 'V_map.npy'), V_learned.reshape(H, W))

        # Reward correlations (Pearson + rank/Spearman)
        rc_all   = reward_correlation(true_reward, learned_reward_flat, mask=nonterminal_mask)
        rcr_all  = reward_rank_correlation(true_reward, learned_reward_flat, mask=nonterminal_mask)
        rc_train = reward_correlation(true_reward, learned_reward_flat, mask=train_mask)
        rcr_train= reward_rank_correlation(true_reward, learned_reward_flat, mask=train_mask)
        rc_test  = reward_correlation(true_reward, learned_reward_flat, mask=test_mask) if test_mask is not None else np.nan
        rcr_test = reward_rank_correlation(true_reward, learned_reward_flat, mask=test_mask) if test_mask is not None else np.nan

        # Value correlations (capture shaping-invariant fidelity)
        def _masked_corr(x, y, m):
            if m is None: return np.nan
            xm, ym = x[m], y[m]
            if xm.size == 0 or ym.size == 0: return np.nan
            if np.std(xm) == 0 or np.std(ym) == 0: return np.nan
            return pearsonr(xm, ym)[0]
        V_corr_all   = _masked_corr(V_true, V_learned, np.ones_like(nonterminal_mask, dtype=bool))
        V_corr_train = _masked_corr(V_true, V_learned, train_mask)
        V_corr_test  = _masked_corr(V_true, V_learned, test_mask) if test_mask is not None else np.nan

        try:
            Pa_list = _extract_Pa(T, env.n_states, env.n_actions)
            Ppi_learned = _build_Ppi(Pa_list, pi_learned)
            # Use uniform over non-terminals as start distribution (robust default)
            d0 = np.zeros(env.n_states, dtype=float)
            d0[nonterminal_mask] = 1.0
            d0_sum = d0.sum()
            if d0_sum > 0:
                d0 /= d0_sum
            occ = _discounted_occupancy(Ppi_learned, gamma, d0)
            # Masks → re-normalised weights
            w_all = occ
            w_train = occ * (train_mask if train_mask is not None else 1.0)
            w_test  = occ * (test_mask  if test_mask  is not None else 0.0)
            # Normalise if non-zero
            if w_train.sum() > 0: w_train = w_train / w_train.sum()
            if w_test.sum()  > 0: w_test  = w_test  / w_test.sum()
            V_corr_w_all   = _weighted_pearson(V_true, V_learned, w_all)
            V_corr_w_train = _weighted_pearson(V_true, V_learned, w_train) if w_train.sum() > 0 else np.nan
            V_corr_w_test  = _weighted_pearson(V_true, V_learned, w_test)  if w_test.sum()  > 0 else np.nan

            agree_vec = (pi_true_actions == pi_learned_actions).astype(float)
            # zero out terminals to avoid scoring actions where policy doesn't matter
            w_all_nt = w_all * nonterminal_mask
            w_train_nt = w_train * (train_mask if train_mask is not None else 1.0)
            w_test_nt  = w_test  * (test_mask  if test_mask  is not None else 0.0)
            pa_w_all   = _weighted_rate_bool(agree_vec, w_all_nt)   if w_all_nt.sum()   > 0 else np.nan
            pa_w_train = _weighted_rate_bool(agree_vec, w_train_nt) if w_train_nt.sum() > 0 else np.nan
            pa_w_test  = _weighted_rate_bool(agree_vec, w_test_nt)  if w_test_nt.sum()  > 0 else np.nan

        except Exception as _e:
            print(f"[WARN] Occupancy-weighted value correlation failed: {_e}")
            V_corr_w_all = V_corr_w_train = V_corr_w_test = np.nan
            pa_w_all = pa_w_train = pa_w_test = np.nan

        results.update({
            "reward_correlation": rc_all,
            "reward_corr_train": rc_train,
            "reward_corr_test": rc_test,
            "reward_spearman": rcr_all,
            "reward_spearman_train": rcr_train,
            "reward_spearman_test": rcr_test,
            "value_difference": np.mean(np.abs(V_true - V_learned)),
            "value_diff_train": np.mean(np.abs(V_true[train_mask] - V_learned[train_mask])) if train_mask is not None else np.nan,
            "value_diff_test": np.mean(np.abs(V_true[test_mask] - V_learned[test_mask])) if test_mask is not None else np.nan,
            "value_correlation": V_corr_all,
            "value_correlation_train": V_corr_train,
            "value_correlation_test": V_corr_test,
            "value_correlation_weighted": V_corr_w_all,
            "value_correlation_weighted_train": V_corr_w_train,
            "value_correlation_weighted_test": V_corr_w_test,
            "policy_agreement": (pi_true_actions == pi_learned_actions).mean(),
            "policy_agreement_train": (pi_true_actions[train_mask] == pi_learned_actions[train_mask]).mean() if train_mask is not None else np.nan,
            "policy_agreement_test": (pi_true_actions[test_mask] == pi_learned_actions[test_mask]).mean() if test_mask is not None else np.nan,
            "policy_agreement_weighted": pa_w_all,
            "policy_agreement_weighted_train": pa_w_train,
            "policy_agreement_weighted_test": pa_w_test,
            "continuous": False
        })

    # Check if this is a continuous environment (like CartPole)
    elif hasattr(env, 'observation_space') and len(env.observation_space.shape) > 0:
        # This is likely a continuous environment
        # Load the learned policy if available
        policy_path = os.path.join(save_dir, 'policy.pt')
        model_weights_path = os.path.join(save_dir, 'model_weights.pt')

        if os.path.exists(policy_path):
            # Reconstruct the policy architecture
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            policy = PolicyNet(state_dim, action_dim)
            policy.load_state_dict(torch.load(policy_path, map_location='cpu'))
            policy.eval()

            return evaluate_continuous_env(env, learned_reward, policy, cfg)

        elif os.path.exists(model_weights_path):
            # Load from model_weights.pt (CartPole case)
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

def _extract_Pa(T, S: int, A: int):
    """
    Extract action-conditioned transition matrices P_a[s, s'] for all actions a.
    Handles both sparse (S*A, S) and dense (S, A, S) layouts.
    """
    if issparse(T):
        return [T[a*S:(a+1)*S, :].toarray() for a in range(A)]
    # Dense path
    if T.ndim == 3 and T.shape == (S, A, S):
        return [T[:, a, :] for a in range(A)]
    # Fallback: try to reshape (S*A, S) -> (A, S, S)
    Ta = np.asarray(T)
    Ta = Ta.reshape(A, S, S)
    return [Ta[a] for a in range(A)]

def _build_Ppi(Pa_list, pi):
    """
    Build P^pi[s, s'] = sum_a pi[s, a] * P_a[s, s'] in a row-wise, dense fashion.
    """
    S, A = pi.shape[0], pi.shape[1]
    Ppi = np.zeros((S, Pa_list[0].shape[1]), dtype=float)
    for a in range(A):
        # Broadcast pi[:, a] over columns of P_a
        Ppi += (pi[:, a][:, None] * Pa_list[a])
    return Ppi

def _discounted_occupancy(Ppi, gamma: float, d0: np.ndarray, max_steps: int = 1000, tol: float = 1e-10):
    """
    Compute discounted state visitation μ = (1-γ) Σ_t γ^t d_t with d_{t+1} = d_t P^π.
    Iterative, numerically stable; returns μ normalised to sum 1.
    """
    S = Ppi.shape[0]
    mu = np.zeros(S, dtype=float)
    d = d0.astype(float).copy()
    if d.sum() <= 0:
        d = np.ones(S, dtype=float) / S
    d /= d.sum()
    g = 1.0
    for t in range(max_steps):
        mu += (1.0 - gamma) * g * d
        d_next = d @ Ppi
        if np.max(np.abs(d_next - d)) < tol and g * gamma < 1e-12:
            d = d_next
            break
        d = d_next
        g *= gamma
        if g < 1e-12:
            break
    s = mu.sum()
    if s > 0:
        mu /= s
    return mu

def _weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """
    Weighted Pearson correlation. Returns np.nan on degeneracy.
    """
    w = np.asarray(w, dtype=float)
    if w.ndim != 1 or x.ndim != 1 or y.ndim != 1:
        return np.nan
    if x.size == 0 or y.size == 0 or w.size == 0 or x.size != y.size or x.size != w.size:
        return np.nan
    sw = w.sum()
    if sw <= 0:
        return np.nan
    mx = (w * x).sum() / sw
    my = (w * y).sum() / sw
    vx = (w * (x - mx) ** 2).sum() / sw
    vy = (w * (y - my) ** 2).sum() / sw
    if vx <= 1e-18 or vy <= 1e-18:
        return np.nan
    cov = (w * (x - mx) * (y - my)).sum() / sw
    return float(cov / np.sqrt(vx * vy))

def _weighted_rate_bool(flags: np.ndarray, w: np.ndarray) -> float:
    """
    Weighted mean of a boolean/0-1 vector.
    Expects flags.shape == w.shape; returns np.nan if weights sum to 0.
    """
    if flags.size == 0 or w.size == 0 or flags.shape[0] != w.shape[0]:
        return np.nan
    sw = float(w.sum())
    if sw <= 0:
        return np.nan
    return float(np.dot(w, flags.astype(float)) / sw)
