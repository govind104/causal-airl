import numpy as np
import pickle
import gc
import psutil
import sys

from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from typing import Dict, Optional, Tuple, Union
from envs.environments import BaseEnv


def log_memory(tag: str):
    """Log current memory usage in MB"""
    process = psutil.Process()
    mem = process.memory_info()
    print(f"[Memory] {tag}: RSS={mem.rss/(1024**2):.2f}MB, VMS={mem.vms/(1024**2):.2f}MB",
          file=sys.stderr)

def compute_state_visitation_frequencies(
    policy: np.ndarray,
    T: Union[np.ndarray, csr_matrix],
    start_dist: np.ndarray,
    gamma: float = 0.99
) -> np.ndarray:
    """
    Computes discounted state visitation frequencies μ^π for a tabular policy.
    Supports both dense and sparse transition matrices.
    """
    if isinstance(T, csr_matrix):
        # Sparse matrix handling
        n_states = T.shape[1]
        n_actions = policy.shape[1]
        policy_flat = policy.flatten()
        
        # Weight T by policy: T_pi = diag(policy_flat) @ T
        T_pi = diags(policy_flat) @ T
        
        # Create reduction matrix R: (n_states, n_states * n_actions)
        rows = np.repeat(np.arange(n_states), n_actions)
        cols = np.arange(n_states * n_actions)
        data = np.ones(n_states * n_actions)
        R = csr_matrix((data, (rows, cols)), shape=(n_states, n_states * n_actions))
        
        # P_π = R * T_pi
        P_pi = R @ T_pi
        
        # Solve: μ = (I - γ P_pi^T)^{-1} @ start_dist
        A = eye(n_states) - gamma * P_pi.T
        return spsolve(A, start_dist)
    else:
        # Dense matrix handling
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

# Warning: linprog(highs-ipm) may fail for large n_states > 10,000
# Consider iterative LP solvers or dual methods for scalability.
def projection_irl(
    T: Union[np.ndarray, csr_matrix],
    expert_policy: np.ndarray,
    start_dist: np.ndarray,
    gamma: float = 0.99,
    l1_norm_bound: float = 1.0,
    save_path: Optional[str] = None,
    use_sparse: bool = True
) -> Tuple[np.ndarray, Dict]:
    # Get dimensions based on T type
    if isinstance(T, csr_matrix):
        n_states = T.shape[1]
        n_actions = T.shape[0] // n_states
    else:
        n_states, n_actions, _ = T.shape
    n = n_states

    # Transition matrix validation
    if isinstance(T, csr_matrix):
        row_sums = T.sum(axis=1).A1
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            diff = np.abs(row_sums - 1.0).max()
            raise ValueError(f"Transition matrix not stochastic. Max deviation: {diff}")
    else:
        if not np.allclose(T.sum(axis=2), 1.0, atol=1e-6):
            diff = np.abs(T.sum(axis=2) - 1.0).max()
            raise ValueError(f"Transition matrix not stochastic. Max deviation: {diff}")

    # 1. Compute state visitation frequencies μ_E
    mu_E = compute_state_visitation_frequencies(expert_policy, T, start_dist, gamma)
    log_memory("After computing state visitation")

    # 2. LP setup: variables [R⁺, R⁻, V] (size 3n)
    #    R = R⁺ - R⁻, R⁺≥0, R⁻≥0, V free
    c = np.zeros(3 * n)
    c[:n] = -mu_E     # R⁺ coefficients
    c[n:2*n] = mu_E   # R⁻ coefficients (since R = R⁺ - R⁻)


    # 3. Constraints in COO format (triplets: data, row, col)
    #    - Expert equality constraints: n equations
    #    - Optimality inequalities: n*(n_actions-1)
    #    - L1 constraint: 1 inequality
    eq_data, eq_rows, eq_cols = [], [], []
    ub_data, ub_rows, ub_cols = [], [], []
    b_eq = []
    b_ub = []

    # Initialize constraint counters
    eq_count = 0
    ub_count = 0

    # Expert policy equality constraints: V(s) = R(s) + γE[V(s')|π_E]
    for s in range(n_states):
        expert_a = np.argmax(expert_policy[s])
        t_row = s * n_actions + expert_a
        
        # Add R⁺[s], R⁻[s], V[s] terms
        eq_data.extend([1.0, -1.0, -1.0])
        eq_rows.extend([eq_count]*3)
        eq_cols.extend([s, n + s, 2*n + s])
        
        # Add transition terms
        if isinstance(T, csr_matrix):
            row = T.getrow(t_row)
            for col, val in zip(row.indices, row.data):
                if abs(val) > 1e-8:
                    eq_data.append(gamma * val)
                    eq_rows.append(eq_count)
                    eq_cols.append(2*n + col)
        else:
            for s_next in range(n_states):
                val = T[s, expert_a, s_next]
                if abs(val) > 1e-8:
                    eq_data.append(gamma * val)
                    eq_rows.append(eq_count)
                    eq_cols.append(2*n + s_next)
                    
        b_eq.append(0.0)
        eq_count += 1

    # Optimality inequalities
    for s in range(n_states):
        expert_a = np.argmax(expert_policy[s])
        for a in range(n_actions):
            if a == expert_a:
                continue
            t_row = s * n_actions + a
            
            # Add R⁺[s], R⁻[s], V[s] terms
            ub_data.extend([1.0, -1.0, -1.0])
            ub_rows.extend([ub_count]*3)
            ub_cols.extend([s, n + s, 2*n + s])
            
            # Add transition terms
            if isinstance(T, csr_matrix):
                row = T.getrow(t_row)
                for col, val in zip(row.indices, row.data):
                    if abs(val) > 1e-8:
                        ub_data.append(gamma * val)
                        ub_rows.append(ub_count)
                        ub_cols.append(2*n + col)
            else:
                for s_next in range(n_states):
                    val = T[s, a, s_next]
                    if abs(val) > 1e-8:
                        ub_data.append(gamma * val)
                        ub_rows.append(ub_count)
                        ub_cols.append(2*n + s_next)
                        
            b_ub.append(0.0)
            ub_count += 1

    # L1 constraint
    for i in range(n):
        ub_data.extend([1.0, 1.0])
        ub_rows.extend([ub_count, ub_count])
        ub_cols.extend([i, n + i])
    b_ub.append(l1_norm_bound)
    ub_count += 1

    # Convert to sparse matrices
    A_eq = coo_matrix((eq_data, (eq_rows, eq_cols)), shape=(eq_count, 3*n)).tocsr()
    A_ub = coo_matrix((ub_data, (ub_rows, ub_cols)), shape=(ub_count, 3*n)).tocsr()
    
    # Clean up constraint data
    del eq_data, eq_rows, eq_cols, ub_data, ub_rows, ub_cols
    gc.collect()
    log_memory("After constraint construction")

    # 4. Bounds
    bounds = [(0, None)] * (2 * n) + [(None, None)] * n

    # 5. Solve LP
    log_memory("Before LP solve")
    res = linprog(
        c=c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method="highs-ipm",
        options={"sparse": True, "presolve": True}
    )
    log_memory("After LP solve")

    # 6. Extract solution
    if res.success:
        R_plus = res.x[:n]
        R_minus = res.x[n:2*n]
        reward = R_plus - R_minus
    else:
        print(f"LP failed with status: {res.status}")
        reward = np.zeros(n)

    # Clean up solver results
    del res
    gc.collect()

    if save_path is not None:
        save_reward(reward, save_path)

    return reward, {"success": res.success, "status": res.message}

def run_ng_russell(
    cfg: dict,
    env: BaseEnv,
    demos: list
) -> Tuple[np.ndarray, dict, dict]:
    """Ng-Russell with sparse matrix support"""
    use_sparse = cfg['irl'].get('use_sparse_constraints', True)
    T = env.build_transition_matrix(sparse=use_sparse)
    
    start_dist = np.ones(env.n_states) / env.n_states
    expert_policy = env.get_optimal_policy()
    
    # Clean up demos
    del demos
    gc.collect()
    
    reward, meta = projection_irl(
        T, expert_policy, start_dist,
        gamma=cfg['irl']['gamma'],
        l1_norm_bound=cfg['irl']['l1_bound'],
        use_sparse=use_sparse
    )
    gc.collect()
    
    return reward, meta, {'reward': reward, 'T': T}