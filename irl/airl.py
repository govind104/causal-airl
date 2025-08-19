import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional, Callable, Union

from envs.environments import BaseEnv
from experiments.logger import TrainingLogger
from models.policy import PolicyNet


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer_encoded_dim(state_encoder, env=None):
    """
    Infer encoded state dimension using actual environment context.
    This avoids ambiguities between GridWorld and CartPole inputs.
    """
    if hasattr(env, "observation_space"):  # Gym-style envs like CartPole
        dummy = torch.zeros(1, env.observation_space.shape[0])
    elif hasattr(env, "grid_size"):  # GridWorld envs
        dummy = torch.zeros(1, 1, dtype=torch.long)  # index form
    else:
        raise ValueError("Cannot infer input shape without known env type.")

    try:
        out = state_encoder(dummy)
        if out.ndim != 2:
            raise ValueError("Encoded output must be 2D")
        return out.shape[1]
    except Exception as e:
        raise ValueError(f"Failed to infer encoded dimension: {e}")

class AIRLDiscriminator(nn.Module):
    """
    Improved AIRL-style discriminator with:
    - Proper expert log_prob handling
    - Stabilized reward computation
    - Performance optimizations
    Supports both discrete and continuous states/actions.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        hidden_dim: int = 64,
        state_encoder: Optional[Callable] = None,
        action_encoder: Optional[Callable] = None,
        reward_logit_clip: Optional[float] = None,
        l2_reg: float = 1e-4
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.reward_logit_clip = reward_logit_clip
        self.l2_reg = l2_reg

        # Encoder functions for custom state/action representations
        self.state_encoder = state_encoder or (lambda x: x)
        self.action_encoder = action_encoder or (lambda x: x)

        # Reward network r(s,a)
        self.r = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Shaping network h(s)
        self.h = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize weights orthogonally for stability
        self._init_weights()

        self.device = get_device()
        self.to(self.device)

    def _init_weights(self):
        for net in [self.r, self.h]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)

    def __call__(self, s, a, s_prime, log_pi=None):
        """
        Unified discriminator interface.

        If log_pi is None:
            Returns f(s,a,s') — the raw reward logits (used in reward shaping)
        Else:
            Returns D(s,a,s') — probability of being expert action.
        """
        if log_pi is None:
            return self.forward(s, a, s_prime)
        return self.D(s, a, s_prime, log_pi)

    def forward(self, s, a, s_prime):
        """
        Compute f(s,a,s') = r(s,a) + γ h(s') - h(s)
        """
        s_enc = self.state_encoder(s)
        a_enc = self.action_encoder(a)
        s_prime_enc = self.state_encoder(s_prime)

        sa = torch.cat([s_enc, a_enc], dim=-1)
        r_sa = self.r(sa)
        h_s = self.h(s_enc)
        h_sp = self.h(s_prime_enc)

        return r_sa + self.gamma * h_sp - h_s

    def D(self, s, a, s_prime, log_pi):
        """Discriminator output: P(expert|s,a,s')"""
        f = self.forward(s, a, s_prime)
        # Rationale: Clipping f BEFORE sigmoid keeps σ(f) away from {0,1} to avoid BCE saturation + nans.
        # ±10 is a safe default (σ(±10)≈{~0,~1} but finite). Leave None unless you actually see saturation in logs.
        if self.reward_logit_clip is not None:
            f = torch.clamp(f, min=-self.reward_logit_clip, max=self.reward_logit_clip)
        return torch.sigmoid(f - log_pi)

    def reward(self, s, a, s_prime=None, log_pi=None):
        """
        Full AIRL reward: f(s,a,s') - log π(a|s) when log_pi provided
        Otherwise returns f(s,a,s') for visualization
        """
        with torch.no_grad():
            f = self.forward(s, a, s_prime if s_prime is not None else s)
            # Same rationale as before; enable ±10 only when needed.
            if self.reward_logit_clip is not None:
                f = torch.clamp(f, min=-self.reward_logit_clip, max=self.reward_logit_clip)
            return f if log_pi is None else f - log_pi

    def _l2_penalty(self):
        """Compute L2 regularization for all networks"""
        return sum(
            layer.weight.pow(2).sum()
            for net in [self.r, self.h]
            for layer in net
            if isinstance(layer, nn.Linear)
        )

class AIRLAgent:
    """Complete AIRL agent with training and evaluation logic"""
    def __init__(
        self,
        env: BaseEnv,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 3e-4,
        device: torch.device = None,
        state_encoder = None,
        action_encoder = None,
        reward_logit_clip: Optional[float] = None,
    ):
        self.device = device or get_device()
        self.gamma = gamma
        self.reward_logit_clip = reward_logit_clip

        self.state_encoder = state_encoder or (lambda x: x)
        self.action_encoder = action_encoder or (lambda x: x)

        self.discriminator = AIRLDiscriminator(
            state_dim, action_dim, gamma,
            state_encoder=self.state_encoder,
            action_encoder=self.action_encoder,
            reward_logit_clip=self.reward_logit_clip
        ).to(self.device)

        encoded_dim = infer_encoded_dim(self.state_encoder, env)
        self.policy = PolicyNet(encoded_dim, action_dim).to(self.device)
        
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr
        )
        self.optimizer_pi = torch.optim.Adam(
            self.policy.parameters(), lr=lr
        )
        self.logger = TrainingLogger()

    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        entropy_coef: float = 0.01,
        grad_clip_norm: float = 0.5
    ) -> float:
        """REINFORCE policy update with baseline, advantage normalization, entropy regularization, and gradient clipping"""
        self.policy.train()
        dist = self.policy(states)
        log_probs = dist.log_prob(actions)

        # Ensure scalar-aligned rewards to avoid (B,B) outer-product via broadcasting
        if rewards.ndim == 2 and rewards.shape[1] == 1:
            rewards = rewards.squeeze(-1)
        elif rewards.ndim > 2:
            rewards = rewards.view(rewards.size(0))

        # Baseline: per-batch mean of rewards
        baseline = rewards.mean().detach()
        advantages = rewards - baseline

        # Advantage normalization with epsilon guard
        adv_std = advantages.std(unbiased=False)
        advantages = advantages / (adv_std + 1e-8)
        advantages = advantages.detach()

        # Entropy bonus
        entropy = dist.entropy().mean()
        loss = -(log_probs * advantages).mean() - entropy_coef * entropy
        self.optimizer_pi.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip_norm)
        self.optimizer_pi.step()

        return loss.item()

    def extract_reward(self, env) -> np.ndarray:
        """
        Compute learned reward over representative states.
        Compatible with:
        - Custom GridWorld (via env.get_all_states)
        - Gymnasium CartPole-v1 (via dynamic rollouts)
        """
        self.discriminator.eval()

        idxs = None
        # Enumerate representative states for reward extraction
        if hasattr(env, 'get_all_states'):
            print("[AIRL] Using full GridWorld state enumeration.")
            raw_states = env.get_all_states()
            # Convert (row, col) coords → flat indices to match training representation
            if len(raw_states) > 0 and isinstance(raw_states[0], (tuple, list)) and len(raw_states[0]) == 2:
                try:
                    indices = [env.state_to_index(s) for s in raw_states]  # preferred signature
                except TypeError:
                    # fallback: pass n_cols if required by env
                    if hasattr(env, "grid_size"):
                        _, W = env.grid_size
                        indices = [env.state_to_index(s, n_cols=W) for s in raw_states]
                    else:
                        indices = [env.state_to_index(s) for s in raw_states]
                idxs = np.asarray(indices, dtype=np.int64)
                # shape [n, 1] as expected by state encoder factories that take scalar index
                states_np = idxs.reshape(-1, 1).astype(np.float32)
            else:
                # Already scalar indices or compatible representation
                as_np = np.array(raw_states)
                # If already scalar indices, record them for scattering
                if as_np.ndim == 1:
                    idxs = as_np.astype(np.int64)
                    states_np = as_np.reshape(-1, 1).astype(np.float32)
                else:
                    # Fallback: unique rows (rare)
                    states_np = as_np.astype(np.float32).reshape(-1, as_np.shape[-1])

        else:
            print("[AIRL] Using replay-based state sampling for reward extraction.")

            def collect_states(policy, episodes=10):
                states = []
                H_max = env.spec.max_episode_steps if hasattr(env, "spec") and env.spec else 200

                with torch.no_grad():
                    for _ in range(episodes):
                        s_raw = env.reset()
                        s = s_raw[0] if isinstance(s_raw, tuple) else s_raw
                        for _ in range(H_max):
                            states.append(s)
                            s_tensor = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                            s_encoded = self.state_encoder(s_tensor)
                            dist = policy(s_encoded)
                            a = dist.sample().item()
                            s, _, terminated, truncated, _ = env.step(a)
                            if terminated or truncated:
                                break
                return states

            agent_states = collect_states(self.policy)
            reset_states = []
            for _ in range(50):
                s_raw = env.reset()
                s = s_raw[0] if isinstance(s_raw, tuple) else s_raw
                reset_states.append(s)
            all_states = agent_states + reset_states

            states_np = np.array(all_states, dtype=np.float32)
            states_np = states_np[np.isfinite(states_np).all(axis=1)]
            states_np = np.unique(states_np, axis=0)

            if len(states_np) == 0:
                raise RuntimeError("No valid states collected for reward extraction.")

            print(f"[AIRL] Sampled {len(all_states)} raw states, "
                f"{len(states_np)} unique valid states.")

            # When we come from rollouts in GridWorld, convert to indices for scattering
            if hasattr(env, "grid_size"):
                _, W = env.grid_size
                idxs = np.array([env.state_to_index(tuple(s.tolist()), n_cols=W) for s in states_np], dtype=np.int64)
                states_np = idxs.reshape(-1, 1).astype(np.float32)

        states = torch.tensor(states_np, dtype=torch.float32, device=self.device)
        dummy_actions = torch.zeros(len(states), dtype=torch.long, device=self.device)  # Assumes discrete actions

        with torch.no_grad():
            rewards = self.discriminator.reward(states, dummy_actions, states)

        rewards_np = rewards.squeeze(-1).detach().cpu().numpy()

        # Canonicalize to a flat vector of length env.n_states in index order
        if hasattr(env, "n_states") and idxs is not None:
            out = np.zeros(int(env.n_states), dtype=rewards_np.dtype)
            # If duplicates exist, last assignment wins (benign for identical states)
            out[idxs] = rewards_np
            return out
        # Fallback: if we cannot form idxs, return as-is (should still pass CartPole)
        return rewards_np

    def update_discriminator(
        self,
        expert_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        agent_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_size: int = 64,
        epochs: int = 3,
        device: torch.device = None
    ) -> Dict[str, float]:
        """
        Improved discriminator training with:
        - Proper expert log_prob computation
        - L2 regularization
        - Memory-efficient batching

        Args:
            expert_data: (s, a, s_prime) tuples
            agent_data: (s, a, s_prime, log_pi) tuples
        """
        if device is None:
            device = self.device

        s_e, a_e, s_prime_e = [x.to(device) for x in expert_data]
        s_pi, a_pi, s_prime_pi, log_pi_agent = [x.to(device) for x in agent_data]

        # Compute expert log probs using current policy
        with torch.no_grad():
            s_e_encoded = self.discriminator.state_encoder(s_e)
            dist_e = self.policy(s_e_encoded)
            log_pi_e = dist_e.log_prob(a_e).unsqueeze(1)  # (B,) -> (B,1)

        # Create dataset
        expert_labels = torch.ones(len(s_e), 1, device=device)
        agent_labels = torch.zeros(len(s_pi), 1, device=device)

        log_pi_e = log_pi_e.detach()
        log_pi_agent = log_pi_agent.detach()

        all_s = torch.cat([s_e, s_pi])
        all_a = torch.cat([a_e, a_pi])
        all_s_prime = torch.cat([s_prime_e, s_prime_pi])
        all_log_pi = torch.cat([log_pi_e, log_pi_agent])
        all_labels = torch.cat([expert_labels, agent_labels])

        # Memory-efficient generator
        def batch_generator():
            perm = torch.randperm(len(all_s))
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i+batch_size]
                yield (all_s[idx], all_a[idx], all_s_prime[idx],
                    all_log_pi[idx], all_labels[idx])

        self.discriminator.train()
        total_loss = 0.0
        total_batches = 0

        for _ in range(epochs):
            for batch in batch_generator():
                s_batch, a_batch, s_prime_batch, log_pi_batch, labels_batch = batch

                # Compute discriminator output
                d_pred = self.discriminator(s_batch, a_batch, s_prime_batch, log_pi_batch)

                # Compute loss + L2 regularization
                loss = F.binary_cross_entropy(d_pred, labels_batch)
                loss += self.discriminator.l2_reg * self.discriminator._l2_penalty()

                # Optimization step
                self.optimizer_d.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0) # Gradient clipping
                self.optimizer_d.step()
                # Finite-loss guard (cfg-neutral; Fails fast if scales explode)
                if not torch.isfinite(torch.tensor(loss.item())):
                    raise RuntimeError("Non-finite discriminator loss (AIRL) — check reward/logit scaling")

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / total_batches if total_batches > 0 else float('nan')
        return {"discriminator_loss": avg_loss}

    def train(self,
              cfg: dict,
              env: BaseEnv,
              demos: list,
              heldout_mask: Optional[np.ndarray] = None,
              save_dir: Optional[str] = None
              ) -> Tuple[np.ndarray, dict, dict]:
        """Full training loop for AIRL"""
        wall_time_cum = 0.0

        # Create checkpoint directory if save_dir provided
        if save_dir is not None:
            os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

        # Prepare expert data
        s_e, a_e, s_pe = self._prepare_expert_data(demos, env)
        
        # Training loop
        for it in range(cfg['irl']['max_iters']):
            t0 = time.perf_counter()

            # Save checkpoint every 25% iterations
            if save_dir is not None and (it % max(1, cfg['irl']['max_iters']//4) == 0 or it == cfg['irl']['max_iters'] - 1):
                try:
                    reward_snapshot = self.extract_reward(env)
                    np.save(os.path.join(save_dir, 'checkpoints', f'reward_iter_{it:04d}.npy'), reward_snapshot)
                except Exception as e:
                    print(f"[WARN] Checkpoint save failed at iter {it}: {e}")

            # Collect agent rollouts
            agent_data = self._collect_agent_rollouts(
                env, cfg['train']['batch_size'], heldout_mask=heldout_mask
            )
            s_pi, a_pi, s_ppi, log_pi = self._process_agent_data(agent_data)

            # Lightweight compute/progress logging
            self.logger.log({"env_steps": len(agent_data)})

            # Evaluate CartPole performance periodically
            if hasattr(env, 'observation_space') and it % 10 == 0:
                cartpole_metrics = self._evaluate_cartpole_performance(env)
                self.logger.log(cartpole_metrics)
                self.logger.log({"continuous": True})

            # Update discriminator
            d_metrics = self.update_discriminator(
                (s_e, a_e, s_pe),
                (s_pi, a_pi, s_ppi, log_pi),
                batch_size=cfg['train']['batch_size'],
                epochs=cfg['train']['epochs'],
                device=self.device
            )
            self.logger.log(d_metrics)
            
            # Update policy with learned rewards
            with torch.no_grad():
                s_pi_encoded = self.state_encoder(s_pi)
                rewards = self.discriminator.reward(s_pi, a_pi, s_ppi, log_pi).detach()


            entropy_coef = cfg.get('irl', {}).get('entropy_coef', 0.01)
            grad_clip_norm = cfg.get('irl', {}).get('grad_clip_norm', 0.5)
            policy_loss = self.update_policy(s_pi_encoded, a_pi.squeeze(), rewards, entropy_coef, grad_clip_norm)
            self.logger.log({"policy_loss": policy_loss})
        

            # Convergence diagnostics
            with torch.no_grad():
                pol_norm = sum(p.norm().item() for p in self.policy.parameters())
                disc_norm = sum(p.norm().item() for p in self.discriminator.parameters())
                self.logger.log({
                    "policy_param_norm": pol_norm,
                    "discriminator_param_norm": disc_norm,
                })

            # Time logging
            dt = time.perf_counter() - t0
            wall_time_cum += dt
            self.logger.log({"epoch_time_sec": dt, "wall_time_sec": wall_time_cum})

        # Final evaluation
        learned_reward = self.extract_reward(env)
        return learned_reward, self.logger.get_logs(), {
            'policy': self.policy,
            'discriminator': self.discriminator,
            'state_encoder': self.state_encoder
        }

    def _evaluate_cartpole_performance(self, env, num_episodes=10):
        """Evaluate CartPole policy performance during training"""
        if not hasattr(env, 'observation_space'):
            return {}

        episode_returns = []
        episode_lengths = []

        self.policy.eval()
        for _ in range(num_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            total_return = 0
            steps = 0
            done = False

            while not done and steps < 500:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    dist = self.policy(obs_tensor)
                    action = dist.sample().item()

                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_return += 1 if not done else 0
                steps += 1

            episode_returns.append(total_return)
            episode_lengths.append(steps)

        return {
            "avg_episode_return": np.mean(episode_returns),
            "avg_episode_length": np.mean(episode_lengths),
            "episode_return_std": np.std(episode_returns)
        }

    def _prepare_expert_data(self, demos, env):
        """Convert expert demos to tensors"""
        s_list, a_list, s_prime_list = [], [], []

        for traj in demos:
            for s, a, _, s_prime in traj:
                s_list.append(torch.FloatTensor(s))
                a_list.append(torch.LongTensor([a]))
                s_prime_list.append(torch.FloatTensor(s_prime))

        # Handle discrete GridWorld-style state indices
        if hasattr(env, "grid_size"):
            H = env.grid_size[1]
            s_raw = torch.stack([
                torch.tensor([[env.state_to_index((int(s[0].item()), int(s[1].item())), n_cols=H)]], dtype=torch.float32)
                for s in s_list
            ]).squeeze(1).to(self.device)

            s_prime_raw = torch.stack([
                torch.tensor([[env.state_to_index((int(sp[0].item()), int(sp[1].item())), n_cols=H)]], dtype=torch.float32)
                for sp in s_prime_list
            ]).squeeze(1).to(self.device)

        # Handle continuous state envs (e.g. CartPole)
        else:
            s_raw = torch.stack(s_list).to(self.device)
            s_prime_raw = torch.stack(s_prime_list).to(self.device)

        return s_raw, torch.stack(a_list).squeeze().to(self.device), s_prime_raw

    def _collect_agent_rollouts(self, env, episodes=10, heldout_mask=None):
        """Collect policy rollouts for training"""
        data = []
        self.policy.eval()

        # Determine input representation
        if hasattr(env, 'grid_size'):
            # GridWorld logic
            W, H = env.grid_size
            min_path = (W - 1) + (H - 1)
            H_max = min(int(2.0 * min_path + 5), 100)

            for _ in range(episodes):
                s = env.reset()
                done = False
                steps = 0

                while not done and steps < H_max:
                    s_index = env.state_to_index(s, n_cols=H)

                    # Skip if current state is in held-out region
                    if heldout_mask is not None and heldout_mask[s_index]:
                        break  # Early episode termination

                    s_tensor = torch.tensor([[s_index]], dtype=torch.float32, device=self.device)
                    dist = self.policy(self.state_encoder(s_tensor))
                    a = dist.sample()
                    logp = dist.log_prob(a)

                    s_next, _, done, _ = env.step(a.item())
                    s_next_index = env.state_to_index(s_next, n_cols=H)

                    # Skip transition if next state is in held-out region
                    if heldout_mask is not None and heldout_mask[s_next_index]:
                        break  # Early episode termination

                    s_next_tensor = torch.tensor([[s_next_index]], dtype=torch.float32, device=self.device)
                    data.append((s_tensor.squeeze(0), a.squeeze(), s_next_tensor.squeeze(0), logp.squeeze()))
                    s = s_next
                    steps += 1

        else:
            # CartPole case — use raw float vectors
            if hasattr(env, "spec") and env.spec is not None:
                max_steps = env.spec.max_episode_steps
            else:
                max_steps = 200
            for _ in range(episodes):
                s = env.reset()
                done = False
                steps = 0
                while not done and steps < max_steps:
                    s_tensor = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                    dist = self.policy(s_tensor)
                    a = dist.sample()
                    logp = dist.log_prob(a)
                    s_next, _, terminated, truncated, _ = env.step(a.item())
                    done = terminated or truncated
                    s_next_tensor = torch.tensor(s_next, dtype=torch.float32, device=self.device).unsqueeze(0)
                    data.append((s_tensor.squeeze(0), a.squeeze(), s_next_tensor.squeeze(0), logp.squeeze()))
                    s = s_next
                    steps += 1

        return data

    def _process_agent_data(self, agent_data):
        """Convert rollout data to tensors"""
        s, a, s_prime, logp = map(torch.stack, zip(*agent_data))
        return s, a.squeeze(), s_prime, logp.unsqueeze(1)

# ========== ENCODERS ========== #

def create_onehot_encoder(num_classes: int):
    """Optimized one-hot encoder"""
    def encoder(x: torch.Tensor):
        assert x.dtype in (torch.long, torch.int), f"Expected integer action indices, got {x.dtype}"
        x = x.view(-1)  # Always flatten to 1D index vector
        return F.one_hot(x.long(), num_classes=num_classes).float()
    return encoder

def create_gridworld_encoder(grid_size: int):
    """Optimized gridworld encoder (2x faster)"""
    def encoder(s: torch.Tensor):
        if s.shape[1] == 1:
            idx = s[:, 0].long()
            row = idx // grid_size
            col = idx % grid_size
        elif s.shape[1] == 2:
            row = s[:, 0].long()
            col = s[:, 1].long()
        else:
            raise ValueError(f"Invalid state shape: {s.shape}")
        flat = row * grid_size + col
        return F.one_hot(flat, num_classes=grid_size**2).float()
    return encoder

class CartpoleEncoder(nn.Module):
    def forward(self, s: torch.Tensor): 
        return s.float() # Ensure tensor is float32

def create_cartpole_encoder():
    return CartpoleEncoder()
