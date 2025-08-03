import abc
import numpy as np
import random
import gymnasium as gym
from typing import Any, List, Tuple, Optional, Dict, Union
from scipy.sparse import csr_matrix, coo_matrix


class BaseEnv(abc.ABC):
    """
    Abstract base class for IRL environments.
    Supports tabular and continuous environments with or without latent confounders.
    """
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            self.set_seed(seed)

    @abc.abstractmethod
    def reset(self) -> Any:
        """Reset the environment and return initial observation/state."""
        pass

    @abc.abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        """
        Apply action to environment.
        Returns: next_state, reward, done, info
        """
        pass

    @abc.abstractmethod
    def sample_expert_trajectories(
        self,
        n_trajectories: int,
        optimality: str = "optimal",
        z: Optional[Any] = None
    ) -> List[List[Tuple[Any, int, float, Any]]]:
        """
        Generate expert demonstrations.
        Returns: List of trajectories, each a list of (s, a, r, s') tuples.
        """
        pass

    @abc.abstractmethod
    def render(self, mode="human"):
        """Optional: Visualise environment state."""
        pass

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

class GridWorld(BaseEnv):
    """
    Classic deterministic gridworld environment.
    Used for tabular IRL methods.
    """
    def __init__(
        self,
        grid_size: Tuple[int, int] = (5, 5),
        terminal_states: List[Tuple[int, int]] = [(4, 4)],
        reward_type: str = "sparse",  # "sparse" or "shaped"
        reward_value: float = 1.0,
        gamma: float = 0.99,
        slip_prob: float = 0.0,
        seed: Optional[int] = None
    ):
        super().__init__(seed)
        self.grid_size = grid_size
        self.n_rows, self.n_cols = grid_size
        self.n_states = self.n_rows * self.n_cols
        self.terminal_states = terminal_states
        self.reward_type = reward_type
        self.reward_value = reward_value
        self.gamma = gamma
        self.slip_prob = slip_prob
        self._cached_T = None
        self._cached_T_sparse = None
        for t in terminal_states:
            if not self.in_bounds(t):
                raise ValueError(f"Terminal state {t} is out of bounds.")

        self.actions = [UP, DOWN, LEFT, RIGHT]
        self.n_actions = len(self.actions)
        self.action_map = {
            UP: (-1, 0),
            DOWN: (1, 0),
            LEFT: (0, -1),
            RIGHT: (0, 1)
        }
        self.reset()

    @staticmethod
    def state_to_index(state: Tuple[int, int], n_cols: int) -> int:
        return state[0] * n_cols + state[1]

    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.n_rows and 0 <= pos[1] < self.n_cols

    def reset(self) -> Tuple[int, int]:
        while True:
            s = (np.random.randint(self.n_rows), np.random.randint(self.n_cols))
            if s not in self.terminal_states:
                self.agent_pos = s
                return s

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        if np.random.rand() < self.slip_prob:
            action = np.random.choice(self.actions)
        delta = self.action_map[action]
        next_pos = (self.agent_pos[0] + delta[0], self.agent_pos[1] + delta[1])
        if not self.in_bounds(next_pos):
            next_pos = self.agent_pos

        reward = self.compute_reward(next_pos)
        done = next_pos in self.terminal_states
        self.agent_pos = next_pos
        return next_pos, reward, done, {}

    def compute_reward(self, pos: Tuple[int, int]) -> float:
        if self.reward_type == "sparse":
            return float(self.reward_value) if pos in self.terminal_states else 0.0
        elif self.reward_type == "shaped":
            goal = self.terminal_states[0]
            dist = np.linalg.norm(np.array(pos) - np.array(goal), ord=1)
            return float(-dist)
        else:
            raise ValueError(f"Unknown reward_type {self.reward_type}")

    def _value_iteration(self, threshold=1e-4) -> np.ndarray:
        V = np.zeros((self.n_rows, self.n_cols))
        policy = np.zeros((self.n_rows, self.n_cols), dtype=int)

        while True:
            delta = 0
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    s = (i, j)
                    if s in self.terminal_states:
                        continue
                    q_vals = []
                    for a in self.actions:
                        delta_pos = self.action_map[a]
                        next_s = (i + delta_pos[0], j + delta_pos[1])
                        if not self.in_bounds(next_s):
                            next_s = s
                        r = self.compute_reward(next_s)
                        q_vals.append(r + self.gamma * V[next_s])
                    best_q = max(q_vals)
                    best_a = np.argmax(q_vals)
                    delta = max(delta, abs(V[s] - best_q))
                    V[s] = best_q
                    policy[s] = best_a
            if delta < threshold:
                break
        return policy

    def sample_expert_trajectories(
        self,
        n_trajectories: int,
        optimality: str = "optimal",
        z: Optional[any] = None
    ) -> List[List[Tuple[Tuple[int, int], int, float, Tuple[int, int]]]]:
        policy = self._value_iteration() if optimality == "optimal" else None
        trajectories = []

        # Compute grid-dependent H_max
        W, H = self.grid_size
        min_path = (W - 1) + (H - 1)
        H_max = min(int(2.0 * min_path + 5), 100)

        for _ in range(n_trajectories):
            traj = []
            s = self.reset()
            done = False
            steps = 0
            if _ == 0:
                print(f"[GridWorld] H_max set to {H_max} for grid size {W}×{H}")
            while not done and steps < H_max:
                a = policy[s] if optimality == "optimal" else np.random.choice(self.actions)
                s_prime, r, done, _ = self.step(a)
                traj.append((s, a, r, s_prime))
                s = s_prime
                steps += 1
            trajectories.append(traj)
        return trajectories

    def render(self, mode="human"):
        grid = np.full((self.n_rows, self.n_cols), ".")
        for t in self.terminal_states:
            grid[t] = "G"
        r, c = self.agent_pos
        grid[r, c] = "A"
        print("\\n".join(" ".join(row) for row in grid))
        print()

    def get_ground_truth_reward(self) -> np.ndarray:
        reward_map = np.zeros((self.n_rows, self.n_cols))
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                reward_map[i, j] = self.compute_reward((i, j))
        return reward_map

    def get_optimal_policy(self) -> np.ndarray:
        """
        Returns policy as array of shape (n_rows * n_cols,), with action index for each state.
        Uses same value iteration logic as expert demo generator.
        """
        policy = self._value_iteration()
        return policy.reshape(-1)

    def get_feature_matrix(self):
        """
        Returns a one-hot feature matrix of shape (n_states, n_states),
        where each row is a one-hot vector for a unique grid cell.
        """
        return np.eye(self.n_states)

    def get_empirical_feature_expectation(self, trajectories):
        """
        Computes empirical state visitation frequency from expert trajectories.

        Returns:
            A 1D numpy array of length n_states, normalized over number of demos.
        """
        feat_exp = np.zeros(self.n_states)
        for traj in trajectories:
            for (s, a, r, s_next) in traj:
                idx = s[0] * self.n_cols + s[1]
                feat_exp[idx] += 1
        return feat_exp / len(trajectories)

    def get_all_states(self):
        return [
            np.array([i, j], dtype=np.float32)
            for i in range(self.n_rows)
            for j in range(self.n_cols)
        ]

    def build_transition_matrix(self, sparse: bool = True) -> Union[np.ndarray, csr_matrix]:
        expected_shape = (self.n_states * self.n_actions, self.n_states)
        if sparse:
            if self._cached_T_sparse is not None:
                if self._cached_T_sparse.shape == expected_shape:
                    return self._cached_T_sparse
                else:
                    print("[Warning] Cached sparse T has wrong shape — rebuilding.")
            # Build sparse COO matrix
            data = []
            rows = []
            cols = []

            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    s = self.state_to_index((i, j), self.n_cols)
                    if (i, j) in self.terminal_states:
                        for a in range(self.n_actions):
                            data.append(1.0)
                            rows.append(s * self.n_actions + a)
                            cols.append(s)
                        continue
                    for a in range(self.n_actions):
                        probs = np.full(self.n_actions, self.slip_prob / (self.n_actions - 1))
                        probs[a] = 1.0 - self.slip_prob
                        for a_prime, p in enumerate(probs):
                            delta = self.action_map[a_prime]
                            ni, nj = i + delta[0], j + delta[1]
                            s_prime = self.state_to_index((ni, nj), self.n_cols) if self.in_bounds((ni, nj)) else s
                            # Only store non-zero probabilities
                            if p > 1e-8:
                                data.append(p)
                                rows.append(s * self.n_actions + a)
                                cols.append(s_prime)
            # Create sparse CSR matrix
            T_sparse = csr_matrix(
                (data, (rows, cols)),
                shape=(self.n_states * self.n_actions, self.n_states)
            )
            self._cached_T_sparse = T_sparse
            return T_sparse
        else:
            if self._cached_T is not None:
                if self._cached_T.shape == (self.n_states, self.n_actions, self.n_states):
                    return self._cached_T
                else:
                    print("[Warning] Cached dense T has wrong shape — rebuilding.")
            # Build dense matrix (original behavior)
            T = np.zeros((self.n_states, self.n_actions, self.n_states))
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    s = self.state_to_index((i, j), self.n_cols)
                    if (i, j) in self.terminal_states:
                        for a in range(self.n_actions):
                            T[s, a, s] = 1.0
                        continue
                    for a in range(self.n_actions):
                        probs = np.full(self.n_actions, self.slip_prob / (self.n_actions - 1))
                        probs[a] = 1.0 - self.slip_prob
                        for a_prime, p in enumerate(probs):
                            delta = self.action_map[a_prime]
                            ni, nj = i + delta[0], j + delta[1]
                            s_prime = self.state_to_index((ni, nj), self.n_cols) if self.in_bounds((ni, nj)) else s
                            T[s, a, s_prime] += p
            self._cached_T = T
            return T
        print(f"Built transition matrix: shape={T_sparse.shape} (sparse={sparse})")

    def reset_transition_cache(self):
        self._cached_T = None
        self._cached_T_sparse = None

class SlipperyGridWorld(GridWorld):
    """
    GridWorld variant with stochastic transitions.
    Overrides slip_prob to introduce random action noise.
    """
    def __init__(
        self,
        grid_size=(5, 5),
        terminal_states=[(4, 4)],
        reward_type="sparse",
        reward_value=1.0,
        gamma=0.99,
        slip_prob=0.2,  # non-zero slip
        seed=None
    ):
        assert slip_prob > 0.0, "Use standard GridWorld for deterministic transitions."
        super().__init__(
            grid_size=grid_size,
            terminal_states=terminal_states,
            reward_type=reward_type,
            reward_value=reward_value,
            gamma=gamma,
            slip_prob=slip_prob,
            seed=seed
        )

class ConfoundedGridWorld(GridWorld):
    """
    GridWorld where expert behavior is influenced by latent confounder Z.
    Z affects expert action choices but is unobserved by standard learners.
    """
    def __init__(
        self,
        grid_size=(5, 5),
        terminal_states=[(4, 4)],
        reward_type="sparse",
        reward_value=1.0,
        gamma=0.99,
        slip_prob=0.0,
        confounder_value: Optional[int] = None,  # e.g., risk-seeking vs. risk-averse
        seed=None
    ):
        super().__init__(
            grid_size=grid_size,
            terminal_states=terminal_states,
            reward_type=reward_type,
            reward_value=reward_value,
            gamma=gamma,
            slip_prob=slip_prob,
            seed=seed
        )
        self.default_z = confounder_value
        self.z: Optional[int] = confounder_value  # current latent confounder

    def _confounded_policy(self, s: Tuple[int, int], z: int) -> int:
        """
        Confounder-conditioned expert policy.
        - z = 0: bias towards goal directly
        - z = 1: prefers bottom-left or risk-averse paths
        """
        r, c = s
        goal_r, goal_c = self.terminal_states[0]

        # Heuristic bias
        if z == 0:  # direct goal-seeker
            delta_r = goal_r - r
            delta_c = goal_c - c
        elif z == 1:  # risk-averse, avoid top-right
            delta_r = -r
            delta_c = -c

        probs = np.ones(4)
        if delta_r < 0: probs[0] += 1  # up
        if delta_r > 0: probs[1] += 1  # down
        if delta_c < 0: probs[2] += 1  # left
        if delta_c > 0: probs[3] += 1  # right
        probs = probs / probs.sum()
        return np.random.choice(self.actions, p=probs)

    def _rollout_confounded_policy(
        self,
        z: int,
        optimality: str = "optimal"
    ) -> List[Tuple[Tuple[int, int], int, float, Tuple[int, int]]]:
        """
        Run one trajectory under expert policy conditioned on confounder Z.
        For simplicity, Z biases the direction of motion.
        """
        s = self.reset()
        traj = []
        done = False

        # Compute grid-dependent H_max
        W, H = self.grid_size
        min_path = (W - 1) + (H - 1)
        H_max = min(int(2.0 * min_path + 5), 100)

        steps = 0
        while not done and steps < H_max:
            if optimality == "random":
                a = np.random.choice(self.actions)
            else:
                a = self._confounded_policy(s, z)
            s_prime, r, done, _ = self.step(a)
            traj.append((s, a, r, s_prime))
            s = s_prime
            steps += 1

        return traj

    def sample_confounded_expert_trajectories(
        self,
        n_trajectories: int,
        optimality: str = "optimal",
        z: Optional[int] = None,
        return_z: bool = False
    ) -> Union[
        List[List[Tuple[Tuple[int, int], int, float, Tuple[int, int]]]],
        List[Tuple[int, List[Tuple[Tuple[int, int], int, float, Tuple[int, int]]]]]
    ]:
        """
        Generate expert rollouts conditioned on fixed or given confounder Z.
        - If `z` is provided, it overrides the default_z set in the constructor.
        """
        if z not in [0, 1]:
            raise ValueError(f"Invalid confounder z={z}. Must be 0 or 1.")
        trajectories = []
        for _ in range(n_trajectories):
            current_z = z if z is not None else self.default_z
            assert current_z is not None, "Confounder z must be provided via config or argument."
            self.z = current_z
            traj = self._rollout_confounded_policy(current_z, optimality=optimality)
            if return_z:
                trajectories.append((current_z, traj))
            else:
                trajectories.append(traj)
        return trajectories

    def get_current_confounder(self) -> Optional[int]:
        return self.z  # for logging/debug
    
    def expert_policy(self, obs: np.ndarray, mode: str = "optimal") -> int:
        """Handles neutral angles in theoretical edge cases"""
        angle = obs[2]
        if angle < -0.05:  # Add tolerance
            return 0
        elif angle > 0.05:
            return 1
        return np.random.choice([0, 1])  # Random if neutral
    
class CartPoleWrapper(BaseEnv):
    """
    Wrapper for CartPole-v1 environment to support AIRL and Causal-AIRL.
    """
    def __init__(
        self,
        render_mode: Optional[str] = None,
        confounder_values: Optional[List[float]] = None,  # e.g., pole lengths
        seed: Optional[int] = None
    ):
        super().__init__(seed)
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.env.reset(seed=seed)
        self.confounder_values = confounder_values or [0.5, 1.0]  # Short vs. long pole
        self.z = None

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_obs, reward, done, info

    def _set_cartpole_length(env, length):
        try:
            env.env.length = length
        except AttributeError:
            raise RuntimeError("Unable to set pole length. Gym version mismatch.")

    def set_confounder(self, z: float):
        """
        Set pole length (confounder). Must be called after reset() in Gym.
        """
        if z not in self.confounder_values:
            raise ValueError(f"Invalid z value: {z}")
        self.z = z
        self._set_cartpole_length(self.env, z)

    def sample_expert_trajectories(
        self,
        n_trajectories: int,
        optimality: str = "optimal",
        z: Optional[float] = None
    ) -> List[List[Tuple[np.ndarray, int, float, np.ndarray]]]:
        """
        Generate expert rollouts with a hard-coded or pre-trained policy.
        """
        trajectories = []

        for _ in range(n_trajectories):
            obs = self.reset()
            if z is not None:
                self.set_confounder(z)
            else:
                self.set_confounder(np.random.choice(self.confounder_values))

            traj = []
            done = False

            # Empirically safe cap for CartPole (default max = 500 steps)
            H_max = 250

            steps = 0
            while not done and steps < H_max:
                a = self.expert_policy(obs, mode=optimality)
                next_obs, r, done, _ = self.step(a)
                traj.append((obs, a, r, next_obs))
                obs = next_obs
                steps += 1
            trajectories.append(traj)

        return trajectories

    def expert_policy(self, obs: np.ndarray, mode: str = "optimal") -> int:
        """
        Handcrafted policy: act to reduce pole angle.
        """
        angle = obs[2]
        return 0 if angle < 0 else 1  # Move left if pole leans left

    def render(self, mode="human"):
        self.env.render()

    def get_current_confounder(self) -> Optional[float]:
        return self.z
    
def build_env(cfg: dict) -> BaseEnv:
    """Factory function to create environment based on config"""
    env_cfg = cfg['env']
    name = env_cfg['name']
    seed = cfg['train'].get('seed', None)
    
    if name == "GridWorld":
        return GridWorld(
            grid_size=env_cfg['grid_size'],
            terminal_states=env_cfg['terminal_states'],
            reward_type=env_cfg['reward_type'],
            reward_value=env_cfg['reward_value'],
            gamma=cfg['irl']['gamma'],
            slip_prob=env_cfg.get('slip_prob', 0.0),
            seed=seed
        )
    elif name == "SlipperyGridWorld":
        return SlipperyGridWorld(
            grid_size=env_cfg['grid_size'],
            terminal_states=env_cfg['terminal_states'],
            reward_type=env_cfg['reward_type'],
            reward_value=env_cfg['reward_value'],
            gamma=cfg['irl']['gamma'],
            slip_prob=env_cfg['slip_prob'],
            seed=seed
        )
    elif name == "ConfoundedGridWorld":
        return ConfoundedGridWorld(
            grid_size=env_cfg['grid_size'],
            terminal_states=env_cfg['terminal_states'],
            reward_type=env_cfg['reward_type'],
            reward_value=env_cfg['reward_value'],
            gamma=cfg['irl']['gamma'],
            slip_prob=env_cfg.get('slip_prob', 0.0),
            confounder_value=env_cfg['confounder_value'],
            seed=seed
        )
    elif name == "CartPole":
        return CartPoleWrapper(
            confounder_values=env_cfg['confounder_values'],
            seed=seed
        )
    else:
        raise ValueError(f"Unknown environment: {name}")
