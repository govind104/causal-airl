import os
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional, Callable, Union

from envs.environments import BaseEnv, CartPoleWrapper
from irl.airl import infer_encoded_dim
from experiments.logger import TrainingLogger
from models.policy import PolicyNet


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CausalPrior(nn.Module):
    """Learnable prior with causal structure"""
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self._device_anchor = nn.Parameter(torch.empty(0))

    def forward(self, batch_size):
        device = self._device_anchor.device
        return Normal(torch.zeros(batch_size, self.latent_dim, device=device),
                     torch.ones(batch_size, self.latent_dim, device=device))

class CausalEncoder(nn.Module):
    """Encoder with causal structure awareness"""
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.mu_net = nn.Linear(hidden_dim, latent_dim)
        self.logvar_net = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, s, a, reparameterize=True):
        if s.ndim != 2 or a.ndim != 2:
            raise ValueError(f"[CausalEncoder] Expected 2D tensors. Got s: {s.shape}, a: {a.shape}")
        x = torch.cat([s, a], dim=-1)
        hidden = self.net(x)
        mu = self.mu_net(hidden)
        logvar = self.logvar_net(hidden)
        std = torch.exp(0.5 * logvar)
        if reparameterize:
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z, mu, std

class CausalDiscriminator(nn.Module):
    """Causal-AIRL discriminator with invariance constraints"""
    def __init__(self, state_dim, action_dim, latent_dim, gamma=0.99, invariance_penalty=0.1, reward_logit_clip: Optional[float] = None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.invariance_penalty = invariance_penalty

        # Optional guard to prevent discriminator saturation
        self.reward_logit_clip = reward_logit_clip

        # Reward networks
        self.r_invariant = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.r_causal = nn.Sequential(
            nn.Linear(state_dim + action_dim + latent_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Shaping function (causal)
        self.h = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def f(self, s, a, s_prime, z):
        """
        Causal reward decomposition: f = r_causal + γh(s') - h(s). Returns (total_reward, invariant_reward)
        """
        # Invariant reward (z-independent)
        r_inv = self.r_invariant(s)

        # Causal reward (z-dependent)
        r_causal = self.r_causal(torch.cat([s, a, z], dim=-1))

        # Full reward (invariant + causal)
        r_full = r_inv + r_causal

        # Potential-based shaping
        h_s = self.h(torch.cat([s, z], dim=-1))
        h_sp = self.h(torch.cat([s_prime, z], dim=-1))

        return (r_full + self.gamma * h_sp - h_s), r_inv

    def D(self, f, log_pi):
        # Optional Clamp of reward logit before sigmoid; cfg-gated
        # Rationale: Clamp f BEFORE sigmoid to prevent discriminator saturation; ±10 is a pragmatic guard if needed.
        if getattr(self, "reward_logit_clip", None) is not None:
            f = torch.clamp(f, min=-self.reward_logit_clip, max=self.reward_logit_clip)
        return torch.sigmoid(f - log_pi)

    def invariance_loss(self, s, a, s_p, z_samples):
        """
        Compute variance of full reward across different latent z samples.
        """
        num_samples, batch_size, _ = z_samples.shape
        s_exp = s.unsqueeze(0).expand(num_samples, -1, -1)  # [num_samples, B, s_dim]
        a_exp = a.unsqueeze(0).expand(num_samples, -1, -1)  # [num_samples, B, a_dim]
        s_p_exp = s_p.unsqueeze(0).expand(num_samples, -1, -1)

        # Flatten for batch processing
        s_flat = s_exp.reshape(-1, s.shape[-1])
        a_flat = a_exp.reshape(-1, a.shape[-1])
        s_p_flat = s_p_exp.reshape(-1, s_p.shape[-1])
        z_flat = z_samples.reshape(-1, z_samples.shape[-1])

        # Get rewards
        full_reward, _ = self.f(s_flat, a_flat, s_p_flat, z_flat)  # [num_samples * batch_size, 1]
        full_reward = full_reward.view(num_samples, batch_size, -1)  # [num_samples, batch_size, 1]

        return full_reward.var(dim=0, unbiased=False).mean()  # [batch_size, 1] → scalar

class CausalAIRLAgent:
    """Complete Causal AIRL agent with training and evaluation"""
    def __init__(
        self,
        env: BaseEnv,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 2,
        gamma: float = 0.99,
        invariance_penalty: float = 0.1,
        lr: float = 3e-4,
        device: torch.device = None,
        state_encoder = None,
        action_encoder = None,
        reward_logit_clip: Optional[float] = None,
    ):
        self.env = env
        self.device = device or get_device()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.reward_logit_clip = reward_logit_clip
        self.gamma = gamma
        self.invariance_penalty = invariance_penalty

        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.encoder = CausalEncoder(state_dim, action_dim, latent_dim).to(self.device)
        self.prior = CausalPrior(latent_dim).to(self.device)

        self.discriminator = CausalDiscriminator(
            state_dim, action_dim, latent_dim, gamma, invariance_penalty,
            reward_logit_clip=self.reward_logit_clip
        ).to(self.device)

        # Use encoded dimension for policy to match AIRL
        encoded_dim = infer_encoded_dim(self.state_encoder, env)
        self.policy = PolicyNet(encoded_dim, action_dim).to(self.device)
        
        self.optimizer_d = torch.optim.Adam(
            list(self.discriminator.parameters()) + 
            list(self.encoder.parameters()) +
            list(self.prior.parameters()),
            lr=lr
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

        expected_encoded_dim = infer_encoded_dim(self.state_encoder, self.env)
        if states.shape[1] == expected_encoded_dim:
            states_encoded = states  # Already encoded
        else:
            states_encoded = self.state_encoder(states)  # Raw states, encode them

        dist = self.policy(states_encoded)
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

    def update_causal_discriminator(
        self,
        expert_data: Tuple[torch.Tensor],
        agent_data: Tuple[torch.Tensor],
        current_epoch: int = 0,
        kl_coeff: float = 1e-2,
        kl_warmup_epochs: int = 20,
        kl_clamp_max: float = 10.0,
        inv_coeff: float = 0.0,
        num_z_samples: int = 5,
        beta_in: float = 1.0,
        batch_size: int = 64,
        epochs: int = 3,
        total_epochs: int = 100
    ) -> Dict[str, float]:

        # Outer-iteration-based linear KL warmup schedule
        def compute_kl_coeff_eff(outer_iter: int, kl_coeff_target: float, warmup_outer_iters: int) -> float:
            """Linear ramp from 0 to target over warmup_outer_iters (outer iterations)"""
            progress = min(1.0, outer_iter / max(1, warmup_outer_iters))
            return progress * kl_coeff_target

        # Compute effective KL coefficient once per outer iteration
        kl_coeff_eff = compute_kl_coeff_eff(current_epoch, kl_coeff, kl_warmup_epochs)

        # Unpack data
        s_e, a_e, s_prime_e = [x.to(self.device) for x in expert_data]
        s_pi, a_pi, s_prime_pi, log_pi_agent = [x.to(self.device) for x in agent_data]

        # Encode ALL states consistently before processing
        s_e_encoded = self.state_encoder(s_e)
        s_prime_e_encoded = self.state_encoder(s_prime_e)
        s_pi_encoded = self.state_encoder(s_pi)
        s_prime_pi_encoded = self.state_encoder(s_prime_pi)

        # Compute expert log_probs using current policy
        with torch.no_grad():
            dist_e = self.policy(s_e_encoded)
            log_pi_expert = dist_e.log_prob(a_e.squeeze()).unsqueeze(1)

        # Combine data
        all_s = torch.cat([s_e_encoded, s_pi_encoded])
        all_a = torch.cat([a_e, a_pi])
        all_s_prime = torch.cat([s_prime_e_encoded, s_prime_pi_encoded])
        all_log_pi = torch.cat([log_pi_expert, log_pi_agent])
        labels = torch.cat([
            torch.ones(len(s_e), 1, device=self.device),
            torch.zeros(len(s_pi), 1, device=self.device)
        ])

        dataset = TensorDataset(all_s, all_a, all_s_prime, all_log_pi, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Accumulate metrics across all inner epochs and batches
        total_loss = 0.0
        total_inv_loss = 0.0
        total_kl_raw = 0.0
        total_kl_post = 0.0
        total_bce = 0.0
        total_var_z_reward = 0.0
        total_z_stats = {'z_mean': 0.0, 'z_std': 0.0, 'z_entropy_approx': 0.0}
        batch_count = 0

        for epoch in range(epochs):
            for s, a, s_p, log_pi, y in loader:
                # Detach agent tensors to avoid backprop into policy graph
                s = s.detach().to(self.device)
                a = a.detach().to(self.device)
                s_p = s_p.detach().to(self.device)
                log_pi = log_pi.detach().to(self.device)
                y = y.detach().to(self.device)

                # Single action encoding - reuse for all paths
                a_enc = self.action_encoder(a)

                # Encode latent
                z_kl, mu, std = self.encoder(s, a_enc)
                q_dist = Normal(mu, std)
                p_dist = self.prior(mu.shape[0])
                kl_raw = kl_divergence(q_dist, p_dist).mean()
                kl_post = torch.clamp(kl_raw, min=0.0, max=kl_clamp_max) # Clamp bound is configurable via config (default 10.0)

                # Compute explicit invariance loss (vectorised & detached z)
                if inv_coeff > 0.0:
                    # Sample z from q(z|s,a) using mu,std from KL path with reparameterization
                    eps = torch.randn(num_z_samples, mu.shape[0], mu.shape[-1], device=self.device)
                    z_samples = mu.unsqueeze(0) + std.unsqueeze(0) * eps  # [K, B, Z]
                    z_samples_detached = z_samples.detach()  # STOP-GRAD through z for invariance loss

                    # If K<2 invariance is undefined; treat as zero penalty (stable & consistent)
                    if num_z_samples < 2:
                        inv_var = torch.tensor(0.0, device=self.device)
                    else:
                        # invariance penalty via discriminator helper (keeps graph)
                        inv_var = self.discriminator.invariance_loss(s, a_enc, s_p, z_samples_detached)

                    # Direct variance metric (detached, no-grad)
                    with torch.no_grad():
                        direct_var_z_reward = inv_var.detach().item()
                    total_var_z_reward += direct_var_z_reward

                else:
                    inv_var = torch.tensor(0.0, device=self.device)
                    total_var_z_reward += 0.0

                # For logging only
                inv_loss_val = inv_var.item() if inv_coeff > 0.0 else 0.0

                # Collect z statistics for logging
                with torch.no_grad():
                    z_mean_batch = mu.mean().item()
                    z_std_batch = std.mean().item()

                    # Entropy approximation for diagonal Gaussian: 0.5 * log(2*pi*e * σ²)
                    z_entropy_batch = (0.5 * torch.log(2 * torch.pi * math.e * std.pow(2))).mean().item()

                    total_z_stats['z_mean'] += z_mean_batch
                    total_z_stats['z_std'] += z_std_batch
                    total_z_stats['z_entropy_approx'] += z_entropy_batch

                # Get discriminator outputs
                f_out, _ = self.discriminator.f(s, a_enc, s_p, z_kl)
                d_pred = self.discriminator.D(f_out, log_pi)
                # Guard BCE against exact 0/1 saturation
                d_pred_safe = d_pred.clamp(1e-6, 1 - 1e-6)

                # Compute losses
                disc_bce = F.binary_cross_entropy(d_pred_safe, y)

                # Single invariance penalty: tensor inv_var stays in computational graph
                mi_proxy = mu.pow(2).mean()
                loss = disc_bce + kl_coeff_eff * kl_post + inv_coeff * inv_var - 1e-3 * mi_proxy
                loss_scalar = loss.item()

                # Optimize
                self.optimizer_d.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
                self.optimizer_d.step()
                # finite-loss guard (cfg-neutral; fails fast if scales explode)
                if not torch.isfinite(torch.tensor(loss.item())):
                    raise RuntimeError("Non-finite discriminator loss (Causal-AIRL) — check reward/logit scaling and KL")

                # Accumulate metrics across all batches
                total_loss += loss.item()
                total_inv_loss += inv_loss_val
                total_kl_raw += kl_raw.item()
                total_kl_post += kl_post.item()
                total_bce += disc_bce.item()
                batch_count += 1

        # Average z statistics across all batches
        if batch_count > 0:
            for key in total_z_stats:
                total_z_stats[key] /= batch_count

        # Compute averages using actual batch count and log once per outer iteration
        mean_disc_total_loss = total_loss / batch_count
        mean_disc_bce = total_bce / batch_count
        mean_kl_raw = total_kl_raw / batch_count
        mean_kl_post = total_kl_post / batch_count
        mean_inv_loss = total_inv_loss / batch_count
        mean_var_z_reward = total_var_z_reward / batch_count

        # Single logging call per outer iteration
        self.logger.log({
            "disc_total_loss": mean_disc_total_loss,
            "epoch_kl_raw": mean_kl_raw,
            "epoch_kl_post": mean_kl_post,
            "kl_coeff_eff": kl_coeff_eff,
            "epoch_inv_loss": mean_inv_loss,
            "epoch_var_z_reward": mean_var_z_reward,
            "epoch_disc_bce": mean_disc_bce,
            "z_mean": total_z_stats['z_mean'],
            "z_std": total_z_stats['z_std'],
            "z_entropy_approx": total_z_stats['z_entropy_approx'],
        })


        return {
            "disc_total_loss": mean_disc_total_loss,
            "disc_bce": mean_disc_bce,
            "kl_raw": mean_kl_raw,
            "kl_post": mean_kl_post,
            "kl_coeff_eff": kl_coeff_eff,
            "z_mean": total_z_stats['z_mean'],
            "z_std": total_z_stats['z_std'],
            "z_entropy_approx": total_z_stats['z_entropy_approx'],
            "inv_loss": mean_inv_loss
        }

    def extract_reward_components(self, env) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Wrapper to redirect to extract_reward_components_for_z with default confounder for CausalAIRL
        """
        z_default = env.confounder_values[0] if hasattr(env, 'confounder_values') and env.confounder_values else 0
        return self.extract_reward_components_for_z(env, z_default)

    def extract_reward_components_for_z(self, env, z_value):
        """
        Compute reward components across representative states for a fixed latent z.
        Handles:
        - Custom GridWorld (via env.get_all_states)
        - Gymnasium CartPole-v1 (via dynamic rollouts)
        Returns: (total_reward, invariant_reward, causal_reward)
        """
        self.discriminator.eval()
        self.encoder.eval()
        idxs = None

        if hasattr(env, 'get_all_states'):
            print("[CausalAIRL] Using full GridWorld state enumeration.")
            all_states = env.get_all_states()
            # Convert (row, col) coords → flat indices to match training representation
            if len(all_states) > 0 and isinstance(all_states[0], (tuple, list)) and len(all_states[0]) == 2:
                try:
                    indices = [env.state_to_index(s) for s in all_states]
                except TypeError:
                    if hasattr(env, "grid_size"):
                        _, W = env.grid_size
                        indices = [env.state_to_index(s, n_cols=W) for s in all_states]
                    else:
                        indices = [env.state_to_index(s) for s in all_states]
                idxs = np.asarray(indices, dtype=np.int64)
                raw_states = idxs.reshape(-1, 1).astype(np.float32)
            else:
                as_np = np.array(all_states)
                if as_np.ndim == 1:
                    idxs = as_np.astype(np.int64)
                    raw_states = as_np.reshape(-1, 1).astype(np.float32)
                else:
                    raw_states = as_np.astype(np.float32).reshape(-1, as_np.shape[-1])
        else:
            print("[CausalAIRL] Using replay-based sampling for reward extraction with fixed z.")
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
                            dist = policy(s_tensor)
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

            raw_states = np.array(all_states, dtype=np.float32)
            raw_states = raw_states[np.isfinite(raw_states).all(axis=1)]
            raw_states = np.unique(raw_states, axis=0)

            if len(raw_states) == 0:
                raise RuntimeError("No valid states collected for reward extraction.")

            print(f"[CausalAIRL] Sampled {len(all_states)} raw states, {len(raw_states)} unique valid states.")
            # When coming from rollouts in GridWorld, convert to indices for scattering
            if hasattr(env, "grid_size"):
                _, W = env.grid_size
                idxs = np.array([env.state_to_index(tuple(s.tolist()), n_cols=W) for s in raw_states], dtype=np.int64)
                raw_states = idxs.reshape(-1, 1).astype(np.float32)

        states = self.state_encoder(torch.tensor(raw_states, dtype=torch.float32, device=self.device))
        # dummy_actions = torch.zeros(len(states), dtype=torch.long, device=self.device)
        dummy_actions = torch.arange(self.action_dim, device=self.device).repeat(len(states) // self.action_dim + 1)[:len(states)]
        actions = self.action_encoder(dummy_actions)
        assert actions.shape[1] == self.action_dim, \
            f"[CausalAIRL] Encoded actions shape mismatch: got {actions.shape[1]}, expected {self.action_dim}"
        #
        # z_tensor = torch.full((len(states), self.latent_dim), fill_value=0.0, device=self.device)
        # z_tensor[:, 0] = float(z_value)

        z_tensor = torch.zeros((len(states), self.latent_dim), device=self.device)
        if isinstance(z_value, int) and 0 <= z_value < self.latent_dim:
            # One-hot encoding for discrete z_value
            z_tensor[:, z_value] = 1.0
        elif isinstance(z_value, (list, tuple)):
            z_val_tensor = torch.tensor(z_value, device=self.device, dtype=torch.float32)
            z_tensor[:, :min(len(z_val_tensor), self.latent_dim)] = z_val_tensor[:self.latent_dim]
        else:
            z_tensor[:, 0] = float(z_value)  # fallback

        with torch.no_grad():
            r_inv = self.discriminator.r_invariant(states)
            r_causal = self.discriminator.r_causal(torch.cat([states, actions, z_tensor], dim=-1))
            total = r_inv + r_causal

        total_np  = total.squeeze(-1).detach().cpu().numpy()
        inv_np    = r_inv.squeeze(-1).detach().cpu().numpy()
        causal_np = r_causal.squeeze(-1).detach().cpu().numpy()

        # Canonicalize to flat vectors of length env.n_states in index order
        if hasattr(env, "n_states") and idxs is not None:
            N = int(env.n_states)
            out_total  = np.zeros(N, dtype=total_np.dtype)
            out_inv    = np.zeros(N, dtype=inv_np.dtype)
            out_causal = np.zeros(N, dtype=causal_np.dtype)
            out_total[idxs]  = total_np
            out_inv[idxs]    = inv_np
            out_causal[idxs] = causal_np
            return out_total, out_inv, out_causal

        # Fallback (CartPole / non-index cases)
        return total_np, inv_np, causal_np

    def _q_warmstart(self, cfg: dict, env: BaseEnv) -> None:
        """Optional q_φ warm-start: pretrain encoder as (s,a)→z classifier."""
        warmstart_cfg = cfg.get('irl', {}).get('q_warmstart', {})
        if not warmstart_cfg.get('enabled', False):
            return  # Skip if disabled

        self.logger.log({"q_warmstart_enabled": 1})

        # Config params
        z_num_classes = warmstart_cfg.get('z_num_classes', 2)
        num_trajs_per_z = warmstart_cfg.get('num_trajs_per_z', 5)

        # Collect labeled (s,a,z) data using existing env hooks
        labeled_data = []
        try:
            if hasattr(env, 'sample_confounded_expert_trajectories'):
                # Preferred: get z-labeled trajectories
                for z_val in range(z_num_classes):
                    trajs = env.sample_confounded_expert_trajectories(
                        num_trajs_per_z, z=z_val, return_z=True
                    )
                    for z_true, traj in trajs:
                        for s, a, _, s_prime in traj:
                            labeled_data.append((s, a, z_true))
            elif hasattr(env, 'confounder_values'):
                # Fallback: iterate confounder_values
                for i, z_val in enumerate(env.confounder_values[:z_num_classes]):
                    trajs = env.sample_expert_trajectories(num_trajs_per_z, z=z_val)
                    for traj in trajs:
                        for s, a, _, s_prime in traj:
                            labeled_data.append((s, a, i))  # Map to class index
            else:
                print("[q_warmstart] No labeled confounder data available - skipping")
                return
        except Exception as e:
            print(f"[q_warmstart] Failed to collect labeled data: {e} - skipping")
            return

        if len(labeled_data) == 0:
            print("[q_warmstart] No labeled data collected - skipping")
            return

        # Fully vectorised preparation
        states, actions, z_labels = zip(*labeled_data)

        # Build raw tensors first, then batch-encode
        states_raw = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)  # [N, ...]
        actions_ix = torch.as_tensor(actions, dtype=torch.long, device=self.device)              # [N]
        z_labels_tensor = torch.as_tensor(z_labels, dtype=torch.float32, device=self.device)     # [N]

        # Encode in-batch (no python loops) — do NOT build autograd graph for warm-start inputs
        with torch.no_grad():
            states_enc = self.state_encoder(states_raw).detach()   # [N, Senc]
            actions_enc = self.action_encoder(actions_ix).detach() # [N, Aenc]

        # Create temporary classification head
        if z_num_classes == 2:
            self.q_head = nn.Linear(self.latent_dim, 1).to(self.device)
            # Optional class weighting for binary case
            pos_weight = warmstart_cfg.get('pos_weight')
            if pos_weight is None:
                n_pos = (z_labels_tensor == 1).sum().item()
                n_neg = (z_labels_tensor == 0).sum().item()
                pos_weight = n_neg / max(1, n_pos)  # Auto: N_neg / N_pos
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.q_head = nn.Linear(self.latent_dim, z_num_classes).to(self.device)
            z_labels_tensor = z_labels_tensor.long()  # CE needs long labels
            # Optional class weighting for multi-class
            class_weight = warmstart_cfg.get('class_weight')
            if class_weight is None:
                # Auto: inverse-frequency with length = z_num_classes (handles missing classes)
                counts = torch.bincount(z_labels_tensor, minlength=z_num_classes).float()
                weights = torch.zeros(z_num_classes, device=self.device)
                present = counts > 0
                if present.any():
                    weights[present] = counts.sum() / (z_num_classes * counts[present])
                class_weight = weights
            if class_weight is not None:
                class_weight = torch.tensor(class_weight, dtype=torch.float32, device=self.device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weight)

        # Freeze non-encoder modules and set proper modes
        self.discriminator.eval()
        self.policy.eval()
        self.prior.eval()

        # Also freeze state/action encoders during warm-start to avoid unintended training
        for p in self.state_encoder.parameters(): p.requires_grad = False
        for p in self.action_encoder.parameters(): p.requires_grad = False

        for param in self.discriminator.parameters():
            param.requires_grad = False
        for param in self.policy.parameters():
            param.requires_grad = False
        for param in self.prior.parameters():
            param.requires_grad = False

        # Set encoder + q_head to training mode
        self.encoder.train()
        self.q_head.train()

        # Dedicated optimizer for encoder + q_head only
        warmstart_lr = warmstart_cfg.get('lr') or cfg['irl']['lr']
        warmstart_wd = warmstart_cfg.get('weight_decay', 0.0)
        warmstart_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.q_head.parameters()),
            lr=warmstart_lr, weight_decay=warmstart_wd
        )

        # Grad clipping norm
        grad_clip_norm = cfg['irl'].get('grad_clip_norm', 0.5)

        # Create DataLoader
        batch_size = warmstart_cfg.get('batch_size') or cfg['train']['batch_size']

        # Pin memory only helps for cpu to cuda transfers; no-op for already-on-device tensors
        pin = (self.device.type == 'cuda' and states_enc.device.type == 'cpu')
        dataset = TensorDataset(states_enc, actions_enc, z_labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=pin)

        # Training loop
        epochs = warmstart_cfg.get('epochs', 1)
        for epoch in range(epochs):
            loss_sum = 0.0
            batch_count = 0

            # Collect mu values for Cohen's d
            all_mu = []
            all_z = []

            for s_batch, a_batch, z_batch in dataloader:
                # Forward pass through encoder
                z_sample, mu, std = self.encoder(s_batch, a_batch)

                # Classification loss
                if z_num_classes == 2:
                    logits = self.q_head(mu).squeeze(-1)
                    loss = loss_fn(logits, z_batch)
                else:
                    logits = self.q_head(mu)
                    loss = loss_fn(logits, z_batch)

                # Optimize with grad clipping
                warmstart_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.q_head.parameters()),
                    grad_clip_norm
                )
                warmstart_optimizer.step()

                loss_sum += loss.item()
                batch_count += 1

                # Store for Cohen's d
                all_mu.append(mu.detach())
                all_z.append(z_batch.detach())

            # Manual Cohen's d computation on mu[:,0]
            if len(all_mu) > 0:
                all_mu_cat = torch.cat(all_mu, dim=0)
                all_z_cat = torch.cat(all_z, dim=0)
                mean_d, max_d = self._cohens_d_binary_all_dims(all_mu_cat, all_z_cat)
            else:
                mean_d, max_d = 0.0, 0.0

            # Log per warm-start epoch only
            avg_loss = loss_sum / batch_count if batch_count > 0 else 0.0
            self.logger.log({
                "q_warmstart_loss": avg_loss,
                "q_warmstart_sep": mean_d,         # Mean across dims
                "q_warmstart_sep_mean": mean_d,    # New explicit mean key
                "q_warmstart_sep_max": max_d       # New max key
            })

        # Unfreeze previously frozen modules
        for p in self.state_encoder.parameters(): p.requires_grad = True
        for p in self.action_encoder.parameters(): p.requires_grad = True

        # Cleanup: restore requires_grad, module modes, and delete q_head
        for param in self.discriminator.parameters():
            param.requires_grad = True
        for param in self.policy.parameters():
            param.requires_grad = True
        for param in self.prior.parameters():
            param.requires_grad = True

        # Restore all modules to training mode
        self.discriminator.train()
        self.policy.train()
        self.prior.train()
        self.encoder.train()

        # Delete temp head
        self.q_head = None
        self.logger.log({"q_warmstart_done": 1})

    def _cohens_d_binary_all_dims(self, mu_values: torch.Tensor, z_labels: torch.Tensor) -> Tuple[float, float]:
        """Compute Cohen's d across all latent dimensions with guards."""
        if z_labels.max() == 0:  # All same class
            return 0.0, 0.0

        # Binary case masks
        mask_0 = (z_labels == 0)
        mask_1 = (z_labels == 1) if z_labels.max() >= 1 else ~mask_0

        mu_0 = mu_values[mask_0]  # [n0, latent_dim]
        mu_1 = mu_values[mask_1]  # [n1, latent_dim]

        n0, n1 = len(mu_0), len(mu_1)
        if min(n0, n1) < 2:  # Insufficient samples
            return 0.0, 0.0

        # Means and variances per dimension [latent_dim]
        m0 = mu_0.mean(dim=0)  # [latent_dim]
        m1 = mu_1.mean(dim=0)  # [latent_dim]
        var0 = mu_0.var(dim=0, unbiased=True)  # [latent_dim]
        var1 = mu_1.var(dim=0, unbiased=True)  # [latent_dim]

        # Pooled standard deviation
        pooled_var = ((n0-1)*var0 + (n1-1)*var1) / max(1, (n0+n1-2))
        pooled_std = torch.sqrt(pooled_var)  # [latent_dim]

        # Guard against near-zero variance per dimension
        valid_dims = pooled_std >= 1e-6
        cohens_d = torch.zeros_like(pooled_std)
        cohens_d[valid_dims] = torch.abs(m1[valid_dims] - m0[valid_dims]) / pooled_std[valid_dims]

        mean_d = cohens_d.mean().item()
        max_d = cohens_d.max().item()

        return mean_d, max_d

    def train(self,
              cfg: dict,
              env: BaseEnv,
              demos: list,
              heldout_mask: Optional[np.ndarray] = None,
              save_dir: Optional[str] = None
              ) -> Tuple[np.ndarray, dict, dict]:
        """Full training loop for Causal AIRL"""
        wall_time_cum = 0.0

        # Create checkpoint directory if save_dir provided
        if save_dir is not None:
            os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

        # Optional q_φ warm-start before main training (soft-skip if disabled/unavailable)
        self._q_warmstart(cfg, env)

        # Prepare expert data
        s_e, a_e, s_pe = self._prepare_expert_data(demos)

        # Read config parameters with defaults
        kl_coeff = cfg.get('irl', {}).get('kl_coeff', 1e-2)
        kl_warmup_epochs = cfg.get('irl', {}).get('kl_warmup_epochs', 20)
        kl_clamp_max = cfg.get('irl', {}).get('kl_clamp_max', 10.0)
        inv_coeff = cfg.get('irl', {}).get('inv_coeff', 0.0)
        num_z_samples = cfg.get('irl', {}).get('num_z_samples', 5)
        entropy_coef = cfg.get('irl', {}).get('entropy_coef', 0.01)

        print(f"[CausalAIRL] Config: kl_coeff={kl_coeff}, kl_warmup_epochs={kl_warmup_epochs}, inv_coeff={inv_coeff}, num_z_samples={num_z_samples}, entropy_coef={entropy_coef}")
        
        # Training loop
        for it in range(cfg['irl']['max_iters']):
            t0 = time.perf_counter()

            # Save checkpoint every 25% iterations
            if save_dir is not None and (it % max(1, cfg['irl']['max_iters']//4) == 0 or it == cfg['irl']['max_iters'] - 1):
                try:
                    reward_snapshot, _, _ = self.extract_reward_components(env)
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

            # Update components
            d_metrics = self.update_causal_discriminator(
                (s_e, a_e, s_pe),
                (s_pi, a_pi, s_ppi, log_pi),
                current_epoch=it,
                kl_coeff=kl_coeff,
                kl_warmup_epochs=kl_warmup_epochs,
                kl_clamp_max=kl_clamp_max,
                inv_coeff=inv_coeff,
                num_z_samples=num_z_samples,
                beta_in=1.0,
                batch_size=cfg['train']['batch_size'],
                epochs=cfg['train']['epochs'],
                total_epochs=cfg['irl']['max_iters']
            )
            
            # Update policy with causal rewards
            with torch.no_grad():
                # Encode states before computing rewards (consistency with discriminator)
                s_pi_encoded = self.state_encoder(s_pi)
                a_enc = self.action_encoder(a_pi)
                z, _, _ = self.encoder(s_pi_encoded, a_enc)
                f_rewards, _ = self.discriminator.f(s_pi_encoded, a_enc, self.state_encoder(s_ppi), z)
                if self.reward_logit_clip is not None: # Clamp f before subtracting log_pi
                    f_rewards = torch.clamp(f_rewards, min=-self.reward_logit_clip, max=self.reward_logit_clip)
                rewards = (f_rewards - log_pi).squeeze(-1) # Reward shaping: f(s,a,s',z) - log π(a|s)

            # Log reward statistics
            with torch.no_grad():
                reward_stats = {
                    "reward_min": rewards.min().item(),
                    "reward_mean": rewards.mean().item(),
                    "reward_max": rewards.max().item()
                }
            self.logger.log(reward_stats)

            entropy_coef = cfg.get('irl', {}).get('entropy_coef', 0.01)
            grad_clip_norm = cfg.get('irl', {}).get('grad_clip_norm', 0.5)
            policy_loss = self.update_policy(s_pi_encoded, a_pi, rewards, entropy_coef, grad_clip_norm)
            self.logger.log({"policy_loss": policy_loss})

            # Convergence diagnostics
            with torch.no_grad():
                pol_norm = sum(p.norm().item() for p in self.policy.parameters())
                disc_norm = sum(p.norm().item() for p in self.discriminator.parameters())
                enc_norm = sum(p.norm().item() for p in self.encoder.parameters())
                self.logger.log({
                    "policy_param_norm": pol_norm,
                    "discriminator_param_norm": disc_norm,
                    "encoder_param_norm": enc_norm,
                })

            # Time logging
            dt = time.perf_counter() - t0
            wall_time_cum += dt
            self.logger.log({"epoch_time_sec": dt, "wall_time_sec": wall_time_cum})

        # Extract reward components
        learned_reward, inv_reward, causal_reward = self.extract_reward_components(env)
        per_z_rewards = []
        if getattr(env, "confounder_values", None):
            for z in env.confounder_values:
                r_z, _, _ = self.extract_reward_components_for_z(env, z)
                per_z_rewards.append(r_z)

        reward_var_z = None
        if per_z_rewards:
            def pad_arrays_to_same_shape(arrays):
                """Ensure all arrays have the same shape by padding with NaN"""
                if not arrays:
                    return []

                shapes = [arr.shape for arr in arrays]
                max_len = max(len(s) for s in shapes)
                max_shape = tuple(max(s[i] if i < len(s) else 0 for s in shapes) for i in range(max_len))

                padded_arrays = []
                for arr in arrays:
                    if arr.shape == max_shape:
                        padded_arrays.append(arr)
                    else:
                        # Create padded array filled with NaN
                        padded = np.full(max_shape, np.nan, dtype=np.float64)
                        # Copy original data using appropriate slicing
                        slices = tuple(slice(0, dim) for dim in arr.shape)
                        padded[slices] = arr.astype(np.float64)
                        padded_arrays.append(padded)
                return padded_arrays

            try:
                padded_rewards = pad_arrays_to_same_shape([r.flatten() for r in per_z_rewards])
                if padded_rewards:
                    reward_var_z = float(np.nanvar(np.stack(padded_rewards, axis=0), axis=0).mean())
                else:
                    reward_var_z = None
            except Exception as e:
                print(f"[Warning] Failed to compute reward variance across z: {e}")
                reward_var_z = None

        # Invariance violation analysis
        if len(per_z_rewards) >= 2:
            try:
                # Invariance violation analysis (if >=2 Zs)
                rz0 = np.asarray(per_z_rewards[0]).ravel()
                rz1 = np.asarray(per_z_rewards[1]).ravel()
                diffs = np.abs(rz0 - rz1)
                inv = {
                    "max_violation": float(diffs.max()),
                    "mean_violation": float(diffs.mean()),
                    "top_violation_state_indices": np.argsort(-diffs)[:10].tolist()
                }
                if save_dir is not None:
                    with open(os.path.join(save_dir, "invariance_analysis.json"), "w") as f:
                        json.dump(inv, f, indent=2)
                    print(f"[INFO] Saved invariance analysis to {save_dir}/invariance_analysis.json")

                r0, r1 = per_z_rewards[0].ravel(), per_z_rewards[1].ravel()
                d = np.abs(r0 - r1)
                invariance_analysis = {
                    'invariance_max': float(d.max()),
                    'invariance_mean': float(d.mean()),
                    'invariance_top_states': np.argsort(-d)[:10].tolist(),
                }
                with open(os.path.join(save_dir, 'invariance_analysis.json'), 'w') as f:
                    json.dump(invariance_analysis, f, indent=2)
            except Exception as e:
                print(f"[WARN] Invariance analysis failed: {e}")

        return learned_reward, self.logger.get_logs(), {
            'policy': self.policy,
            'discriminator': self.discriminator,
            'encoder': self.encoder,
            'invariant_reward': inv_reward,
            'causal_reward': causal_reward,
            "per_z_rewards": per_z_rewards,
            "state_encoder": self.state_encoder,
            "reward_var_z": reward_var_z
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

    def _prepare_expert_data(self, demos):
        """Convert expert demos to tensors - KEEP RAW like AIRL"""
        s_list, a_list, s_prime_list = [], [], []

        for traj in demos:
            for s, a, _, s_prime in traj:
                s_list.append(torch.FloatTensor(s))
                a_list.append(torch.LongTensor([a]))
                s_prime_list.append(torch.FloatTensor(s_prime))

        # Handle discrete GridWorld-style state indices like AIRL
        if hasattr(self.env, "grid_size"):
            H = self.env.grid_size[1]
            s_raw = torch.stack([
                torch.tensor([[self.env.state_to_index((int(s[0].item()), int(s[1].item())), n_cols=H)]], dtype=torch.float32)
                for s in s_list
            ]).squeeze(1).to(self.device)

            s_prime_raw = torch.stack([
                torch.tensor([[self.env.state_to_index((int(sp[0].item()), int(sp[1].item())), n_cols=H)]], dtype=torch.float32)
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
