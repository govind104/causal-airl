import numpy as np
import pickle

from typing import Tuple, List, Dict, Optional
from envs.environments import BaseEnv, build_transition_matrix

def soft_value_iteration(
    T: np.ndarray,
    rewards: np.ndarray,
    gamma: float = 0.99,
    threshold: float = 1e-6,
    max_iter: int = 1000
) -> np.ndarray:
    """
    Soft value iteration for MaxEnt IRL with numerical stability.

    Args:
        T: Transition matrix [S, A, S']
        rewards: State rewards R(s)
        gamma: Discount factor
        threshold: Convergence threshold
        max_iter: Maximum iterations

    Returns:
        V: Soft value function
    """
    n_states, n_actions, _ = T.shape
    V = np.zeros(n_states)

    for _ in range(max_iter):
        V_prev = V.copy()
        Q = np.zeros((n_states, n_actions))

        # Vectorized Q computation
        Q = rewards[:, None] + gamma * np.einsum('saS,S->sa', T, V)
        max_Q = np.max(Q, axis=1)

        V = max_Q + np.log(np.sum(np.exp(Q - max_Q[:, None]), axis=1))
        if np.max(np.abs(V - V_prev)) < threshold:
            break
    return V


def compute_policy_from_value(
    T: np.ndarray,
    rewards: np.ndarray,
    V: np.ndarray,
    gamma: float = 0.99
) -> np.ndarray:
    """
    Compute soft-optimal policy from soft value function with numerical stability.

    Returns:
        policy: [S, A] softmax over Q(s,a)
    """
    n_states, n_actions, _ = T.shape
    Q = np.zeros((n_states, n_actions))

    for a in range(n_actions):
        Q[:, a] = rewards + gamma * T[:, a, :] @ V

    # Stable softmax
    max_Q = np.max(Q, axis=1, keepdims=True)
    exp_Q = np.exp(Q - max_Q)
    pi = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)
    return pi


def expected_svf(
    T: np.ndarray,
    policy: np.ndarray,
    start_dist: np.ndarray,
    gamma: float = 0.99
) -> np.ndarray:
    """
    Compute expected state visitation frequencies using exact matrix solution.

    Returns:
        mu: [S] discounted visitation frequencies
    """
    n_states = T.shape[0]
    # Compute policy-weighted transition: P_π[s,s'] = Σ_a π(a|s)T(s,a,s')
    P_pi = np.einsum('sa,saS->sS', policy, T)

    # Solve (I - γP_πᵀ)μ = μ₀
    A = np.eye(n_states) - gamma * P_pi.T
    return np.linalg.solve(A, start_dist)


def maxent_irl(
    feature_matrix: np.ndarray,
    T: np.ndarray,
    trajectories: List[List[int]],
    start_dist: np.ndarray,
    gamma: float = 0.99,
    learning_rate: float = 0.1,
    momentum: float = 0.8,
    n_iters: int = 100,
    tol: float = 1e-5,
    reg_lambda: float = 0.0,
    normalize_features: bool = True,
    verbose: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Enhanced Maximum Entropy IRL with numerical stability and optimization improvements.

    Args:
        feature_matrix: [S, D] feature matrix
        T: transition matrix [S, A, S']
        trajectories: list of expert state sequences
        start_dist: [S] initial state distribution
        gamma: discount factor
        learning_rate: gradient ascent step size
        momentum: momentum coefficient (0 to disable)
        n_iters: maximum optimization iterations
        tol: convergence tolerance (gradient norm)
        reg_lambda: L2 regularization strength
        normalize_features: whether to normalize features
        verbose: print progress

    Returns:
        reward: [S] recovered reward function
        meta: training logs (loss, grad_norm, theta)
    """
    n_states, n_features = feature_matrix.shape
    n_states, n_actions, _ = T.shape

    # Optional feature normalization
    if normalize_features:
        feature_matrix = feature_matrix / (np.linalg.norm(feature_matrix, axis=0) + 1e-8)

    # Compute γ-discounted expert feature expectations
    mu_E = np.zeros(n_features)
    for traj in trajectories:
        discount = 1.0
        for s in traj:
            mu_E += discount * feature_matrix[s]
            discount *= gamma
    mu_E /= len(trajectories)
    if verbose:
        print(f"Expert feature expectations: {mu_E}")

    # Initialize parameters and optimizer state
    theta = np.random.uniform(-0.1, 0.1, size=n_features)
    prev_grad = np.zeros_like(theta)
    logs = {"loss": [], "grad_norm": [], "theta": [theta.copy()]}
    
    best_theta = theta.copy()
    best_loss = float('inf')

    for i in range(n_iters):
        # Compute current reward function
        R = feature_matrix.dot(theta)

        # Compute optimal policy under current reward
        V = soft_value_iteration(T, R, gamma)
        pi = compute_policy_from_value(T, R, V, gamma)

        # Compute expected state visitation frequencies
        mu = expected_svf(T, pi, start_dist, gamma)
        mu_model_feat = feature_matrix.T.dot(mu)

        # Compute gradient with regularization
        grad = mu_E - mu_model_feat - reg_lambda * theta
        grad_norm = np.linalg.norm(grad)

        # Update with momentum
        update = learning_rate * grad + momentum * prev_grad
        theta += update
        prev_grad = update.copy()

        # Track convergence metrics
        loss = 0.5 * np.sum((mu_E - mu_model_feat)**2)  # MSE loss
        logs["loss"].append(loss)
        logs["grad_norm"].append(grad_norm)
        logs["theta"].append(theta.copy())

        if verbose and (i % 10 == 0 or i == n_iters-1):
            print(f"Iter {i:3d}: Loss = {loss:.6f}, Grad = {grad_norm:.6f}")

        # Track best parameters
        if loss < best_loss:
            best_loss = loss
            best_theta = theta.copy()
            
        # Early stopping A
        if i > 10 and abs(loss - logs["loss"][-2]) < tol/10:
            if verbose: print(f"Early stop at iter {i}")
            theta = best_theta
            break

        # Early stopping B
        if grad_norm < tol:
            if verbose:
                print(f"Converged at iter {i}")
            break

    # Final reward function
    reward = feature_matrix.dot(theta)
    return reward, logs


# Helper functions for diagnostics and visualization
def compute_expected_features(
    feature_matrix: np.ndarray,
    trajectories: List[List[int]],
    gamma: float = 0.99
) -> np.ndarray:
    """Compute discounted feature expectations from trajectories"""
    n_features = feature_matrix.shape[1]
    mu_E = np.zeros(n_features)
    for traj in trajectories:
        discount = 1.0
        for s in traj:
            mu_E += discount * feature_matrix[s]
            discount *= gamma
    return mu_E / len(trajectories)


def compute_reward_error(
    true_reward: np.ndarray,
    learned_reward: np.ndarray
) -> Dict[str, float]:
    """Compute reward recovery metrics"""
    return {
        "mse": np.mean((true_reward - learned_reward)**2),
        "corr": np.corrcoef(true_reward, learned_reward)[0,1],
        "l1": np.mean(np.abs(true_reward - learned_reward))
    }

def save_maxent_model(theta: np.ndarray, config: Dict, path: str):
    """
    Save theta parameters and config to file.
    """
    with open(path, "wb") as f:
        pickle.dump({"theta": theta, "config": config}, f)

def load_maxent_model(path: str) -> Tuple[np.ndarray, Dict]:
    """
    Load theta parameters and config from file.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["theta"], data["config"]

def preprocess_demos(env, demos) -> List[List[int]]:
    """Convert expert trajectories to state sequences"""
    trajectories = []
    for traj in demos:
        state_seq = []
        for (s, a, r, s_prime) in traj:
            if hasattr(env, 'state_to_index'):
                state_seq.append(env.state_to_index(s))
            else:  # Handle continuous states
                state_seq.append(s)
        trajectories.append(state_seq)
    return trajectories

def run_maxent_irl(
    cfg: dict,
    env: BaseEnv,
    demos: list
) -> Tuple[np.ndarray, dict, dict]:
    """Unified MaxEnt IRL interface"""
    feature_matrix = env.get_feature_matrix()
    T = build_transition_matrix(env.grid_size, slip_prob=cfg['env'].get('slip_prob', 0.0))
    trajectories = preprocess_demos(env, demos)
    start_dist = np.ones(env.n_states) / env.n_states
    
    reward, logs = maxent_irl(
        feature_matrix, T, trajectories, start_dist,
        gamma=cfg['irl']['gamma'],
        learning_rate=cfg['irl']['lr'],
        n_iters=cfg['irl']['max_iters'],
        verbose=cfg['train']['verbose']
    )
    
    return reward, {'training_logs': logs}, {'reward': reward}