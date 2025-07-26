import numpy as np
import pickle

from scipy.optimize import linprog
from typing import List, Tuple, Dict, Optional
from envs.environments import BaseEnv, build_transition_matrix


def compute_state_visitation_frequencies(
    policy: np.ndarray,
    T: np.ndarray,
    start_dist: np.ndarray,
    gamma: float = 0.99
) -> np.ndarray:
    """
    Computes discounted state visitation frequencies μ^π for a tabular policy.
    """
    n_states, n_actions, _ = T.shape
    P_pi = np.zeros((n_states, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            P_pi[s] += policy[s, a] * T[s, a]
    A = np.eye(n_states) - gamma * P_pi.T
    return np.linalg.solve(A, start_dist)

def save_reward(reward: np.ndarray, path: str):
    with open(path, "wb") as f:
        pickle.dump(reward, f)

def load_reward(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return pickle.load(f)

def projection_irl(
    T: np.ndarray,
    expert_policy: np.ndarray,
    start_dist: np.ndarray,
    gamma: float = 0.99,
    l1_norm_bound: float = 1.0,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, Dict]:

    # Transition matrix validation
    if not np.allclose(T.sum(axis=2), 1.0, atol=1e-6):
        diff = T.sum(axis=2) - 1.0
        raise ValueError(f"Transition matrix not stochastic (Rows don't sum to 1). Max deviation: {np.abs(diff).max()}")

    n_states, n_actions, _ = T.shape
    n = n_states

    # 1. Compute state visitation frequencies μ_E for objective weights
    mu_E = compute_state_visitation_frequencies(expert_policy, T, start_dist, gamma)

    # 2. LP setup: variables [R⁺, R⁻, V] (size 3n)
    #    R = R⁺ - R⁻, R⁺≥0, R⁻≥0, V free
    c = np.zeros(3 * n)
    for i in range(n):
        c[i] = -mu_E[i]      # -R⁺ coefficient
        c[n + i] = mu_E[i]   # -R⁻ coefficient (since R = R⁺ - R⁻)

    # 3. Constraints:
    #    - Expert equality constraints: n equations
    #    - Optimality inequalities: n*(n_actions-1)
    #    - L1 constraint: 1 inequality
    A_eq, b_eq = [], []  # Expert policy equalities
    A_ub, b_ub = [], []  # Optimality inequalities + L1

    # Expert policy equality constraints: V(s) = R(s) + γE[V(s')|π_E]
    for s in range(n):
        expert_action = np.argmax(expert_policy[s])  # Assumes deterministic policy
        row = np.zeros(3 * n)
        # R(s) = R⁺[s] - R⁻[s]
        row[s] = 1          # R⁺[s]
        row[n + s] = -1     # -R⁻[s]
        row[2*n + s] = -1   # -V(s)

        # γΣT(s,a,s')V(s')
        for s_prime in range(n):
            row[2*n + s_prime] += gamma * T[s, expert_action, s_prime]
        A_eq.append(row)
        b_eq.append(0.0)

    # Optimality inequalities: Q(s,a) ≤ V(s) for a ≠ π_E(s)
    for s in range(n):
        expert_action = np.argmax(expert_policy[s])
        for a in range(n_actions):
            if a == expert_action:
                continue
            row = np.zeros(3 * n)
            # R(s) = R⁺[s] - R⁻[s]
            row[s] = 1          # R⁺[s]
            row[n + s] = -1     # -R⁻[s]
            row[2*n + s] = -1   # -V(s)

            # γΣT(s,a,s')V(s')
            for s_prime in range(n):
                row[2*n + s_prime] += gamma * T[s, a, s_prime]
            A_ub.append(row)
            b_ub.append(0.0)

    # L1 constraint: Σ(R⁺[i] + R⁻[i]) ≤ bound
    row_l1 = np.zeros(3 * n)
    row_l1[:2*n] = 1  # Sum all R⁺ and R⁻
    A_ub.append(row_l1)
    b_ub.append(l1_norm_bound)

    # 4. Bounds: R⁺≥0, R⁻≥0, V unbounded
    bounds = [(0, None)] * (2 * n) + [(None, None)] * n

    # 5. Solve LP
    res = linprog(
        c=c,
        A_ub=np.array(A_ub), b_ub=np.array(b_ub),
        A_eq=np.array(A_eq), b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs"
    )

    # 6. Extract solution
    if res.success:
        R_plus = res.x[:n]
        R_minus = res.x[n:2*n]
        reward = R_plus - R_minus
    else:
        reward = np.zeros(n)

    if save_path is not None:
        save_reward(reward, save_path)

    return reward, {"success": res.success, "status": res.message}

def run_ng_russell(
    cfg: dict,
    env: BaseEnv,
    demos: list
) -> Tuple[np.ndarray, dict, dict]:
    """Unified Ng-Russell interface"""
    T = build_transition_matrix(env.grid_size, slip_prob=cfg['env'].get('slip_prob', 0.0))
    start_dist = np.ones(env.n_states) / env.n_states
    expert_policy = env.get_optimal_policy()
    
    reward, meta = projection_irl(
        T, expert_policy, start_dist,
        gamma=cfg['irl']['gamma'],
        l1_norm_bound=cfg['irl']['l1_bound']
    )
    
    return reward, meta, {'reward': reward}