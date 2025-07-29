import numpy as np
import pickle
import gc
import psutil
import sys

from scipy.sparse import issparse, csr_matrix, diags, eye
from scipy.sparse.linalg import spsolve
from typing import Tuple, List, Dict, Union
from envs.environments import BaseEnv


def log_memory(tag: str):
    """Log current memory usage in MB"""
    process = psutil.Process()
    mem = process.memory_info()
    print(f"[Memory] {tag}: RSS={mem.rss/(1024**2):.2f}MB, VMS={mem.vms/(1024**2):.2f}MB",
          file=sys.stderr)
    
def soft_value_iteration(
    T: Union[np.ndarray, csr_matrix],
    rewards: np.ndarray,
    gamma: float = 0.99,
    threshold: float = 1e-6,
    max_iter: int = 1000
) -> np.ndarray:
    """
    Soft value iteration for MaxEnt IRL with numerical stability. Supports sparse/dense computation.

    Args:
        T: Transition matrix [S, A, S']
        rewards: State rewards R(s)
        gamma: Discount factor
        threshold: Convergence threshold
        max_iter: Maximum iterations

    Returns:
        V: Soft value function
    """
    if issparse(T):
        # Sparse matrix handling
        n_states = T.shape[1]
        n_actions = T.shape[0] // n_states
        V = np.zeros(n_states)
        
        for _ in range(max_iter):
            V_prev = V.copy()
            # Compute Q(s,a) = T(s,a) @ (rewards + gamma*V)
            Q_vals = T.dot(rewards + gamma * V)
            Q = Q_vals.reshape(n_states, n_actions)
            max_Q = np.max(Q, axis=1, keepdims=True)
            exp_Q = np.exp(Q - max_Q)
            sum_exp = np.sum(exp_Q, axis=1, keepdims=True)
            # Numerical stability for near-zero probabilities
            sum_exp[sum_exp < 1e-20] = 1e-20
            V = (max_Q + np.log(sum_exp)).flatten()
            
            if np.max(np.abs(V - V_prev)) < threshold:
                break
        return V
    else:
        # Dense matrix handling
        n_states, n_actions, _ = T.shape
        V = np.zeros(n_states)
        
        for _ in range(max_iter):
            V_prev = V.copy()
            Q = np.zeros((n_states, n_actions))
            for a in range(n_actions):
                Q[:, a] = T[:, a, :].dot(rewards + gamma * V)
            max_Q = np.max(Q, axis=1, keepdims=True)
            exp_Q = np.exp(Q - max_Q)
            sum_exp = np.sum(exp_Q, axis=1, keepdims=True)
            # Numerical stability for near-zero probabilities
            sum_exp[sum_exp < 1e-20] = 1e-20
            V = (max_Q + np.log(sum_exp)).flatten()
            if np.max(np.abs(V - V_prev)) < threshold:
                break
        return V


def compute_policy_from_value(
    T: Union[np.ndarray, csr_matrix],
    rewards: np.ndarray,
    V: np.ndarray,
    gamma: float = 0.99
) -> np.ndarray:
    """
    Compute soft-optimal policy from soft value function with numerical stability. Supports sparse/dense computation.

    Returns:
        policy: [S, A] softmax over Q(s,a)
    """
    if issparse(T):
        # Sparse matrix handling
        n_states = V.shape[0]
        n_actions = T.shape[0] // n_states
        Q_vals = T.dot(rewards + gamma * V)
        Q = Q_vals.reshape(n_states, n_actions)
    else:
        # Dense matrix handling
        n_states, n_actions, _ = T.shape
        Q = np.zeros((n_states, n_actions))
        for a in range(n_actions):
            Q[:, a] = T[:, a, :].dot(rewards + gamma * V)
    
    # Stable softmax
    max_Q = np.max(Q, axis=1, keepdims=True)
    exp_Q = np.exp(Q - max_Q)
    sum_exp = np.sum(exp_Q, axis=1, keepdims=True)
    # Prevent division by zero
    sum_exp[sum_exp < 1e-20] = 1e-20
    pi = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)
    return pi


def expected_svf(
    T: Union[np.ndarray, csr_matrix],
    policy: np.ndarray,
    start_dist: np.ndarray,
    gamma: float = 0.99
) -> np.ndarray:
    """
    Compute expected state visitation frequencies using exact matrix solution. Supports sparse/dense computation.

    Returns:
        mu: [S] discounted visitation frequencies
    """
    if issparse(T):
        # Sparse matrix handling
        n_states = T.shape[1]
        n_actions = T.shape[0] // n_states
        
        # Create policy-weighted T: P_π(s,s') = Σ_a π(a|s)T(s,a,s')
        policy_flat = policy.flatten()
        T_weighted = diags(policy_flat) @ T
        
        # Create reduction matrix R: (n_states, n_states * n_actions)
        rows = np.repeat(np.arange(n_states), n_actions)
        cols = np.arange(n_states * n_actions)
        R = csr_matrix((np.ones_like(rows), (rows, cols)), 
                      shape=(n_states, n_states * n_actions))
        
        P_pi = R @ T_weighted
    else:
        # Dense matrix handling
        n_states, n_actions, _ = T.shape
        P_pi = np.einsum('sa,saS->sS', policy, T)
    
    # Solve (I - γP_πᵀ)μ = μ₀
    A = eye(n_states) - gamma * P_pi.T
    return spsolve(A, start_dist)


def maxent_irl(
    feature_matrix: np.ndarray,
    T: Union[np.ndarray, csr_matrix],
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
        
        # Memory cleanup every 10 iterations
        if i % 10 == 0:
            gc.collect()

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
    use_sparse = cfg['irl'].get('use_sparse_constraints', False)
    if not use_sparse and env.n_states > 5000:
        print("[Warning] MaxEnt IRL using dense transitions on large grid — memory risk.")
    T = env.build_transition_matrix(sparse=use_sparse)
    trajectories = preprocess_demos(env, demos)
    start_dist = np.ones(env.n_states) / env.n_states
    
    log_memory("Before MaxEnt IRL")
    reward, logs = maxent_irl(
        feature_matrix, T, trajectories, start_dist,
        gamma=cfg['irl']['gamma'],
        learning_rate=cfg['irl']['lr'],
        momentum=cfg['irl'].get('momentum', 0.8),
        reg_lambda=cfg['irl'].get('reg_lambda', 0.0),
        normalize_features=cfg['irl'].get('normalize_features', True),
        n_iters=cfg['irl']['max_iters'],
        verbose=cfg['train']['verbose']
    )
    log_memory("After MaxEnt IRL")
    
    # Clean up
    del trajectories, demos
    gc.collect()
    
    return reward, {'training_logs': logs}, {'reward': reward, 'T': T}