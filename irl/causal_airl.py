import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional, Callable

from envs.environments import BaseEnv, CartPoleWrapper
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
    def __init__(self, state_dim, action_dim, latent_dim, gamma=0.99, invariance_penalty=0.1):
        super().__init__()
        self.gamma = gamma
        self.invariance_penalty = invariance_penalty

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

        return full_reward.var(dim=0).mean()  # [batch_size, 1] → scalar

    def compute_reward(self, s, a=None):
        """
        Unified reward interface for attribution and slicing.
        Assumes self.state_encoder and self.reward_net exist.
        """
        if hasattr(self, "state_encoder"):
            s = self.state_encoder(s)
        if a is not None:
            a = a if a.ndim == 2 else a.unsqueeze(-1)
            input_tensor = torch.cat([s, a], dim=-1)
            return self.r_causal(input_tensor).squeeze()
        return self.r_invariant(s).squeeze()

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
        action_encoder = None
    ):
        self.env = env
        self.device = device or get_device()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.gamma = gamma
        print(f"[INIT] CausalAIRL latent_dim={latent_dim}, gamma={gamma}")
        self.invariance_penalty = invariance_penalty
        
        self.encoder = CausalEncoder(state_dim, action_dim, latent_dim).to(self.device)
        print(f"[INIT] CausalEncoder expected input dim: {state_dim + action_dim}")
        self.prior = CausalPrior(latent_dim).to(self.device)
        self.discriminator = CausalDiscriminator(
            state_dim, action_dim, latent_dim, gamma, invariance_penalty
        ).to(self.device)
        self.policy = PolicyNet(state_dim, action_dim).to(self.device)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        
        self.optimizer_d = torch.optim.Adam(
            list(self.discriminator.parameters()) + 
            list(self.encoder.parameters()) +
            list(self.prior.parameters()),
            lr=lr
        )
        self.optimizer_pi = torch.optim.Adam([
            {'params': self.policy.parameters()},
            {'params': self.value_net.parameters()}
        ], lr=lr)
        self.logger = TrainingLogger()

    def get_reward(self, s, a, s_prime, z=None):
        """Get causal-aware reward"""
        with torch.no_grad():
            if z is None:
                # Sample latent if not provided
                a = self.action_encoder(a)
                z, _, _ = self.encoder(s, a)
            reward, _ = self.discriminator.f(s, a, s_prime, z)
        return reward

    def get_invariant_reward(self, s):
        """Get Z-invariant reward component"""
        with torch.no_grad():
            reward = self.discriminator.r_invariant(s)
        return reward

    def get_reward_components(self, s, a, s_prime):
        """Returns (total_reward, invariant_component, causal_component)"""
        assert a.ndim == 2, f"[get_reward_components] Expected one-hot actions. Got shape: {a.shape}"
        with torch.no_grad():
            if a.ndim == 3:
                a = a.squeeze(-1)
            z, _, _ = self.encoder(s, a)

            r_inv = self.discriminator.r_invariant(s)
            r_causal = self.discriminator.r_causal(torch.cat([s, a, z], -1))

        return r_inv + r_causal, r_inv, r_causal

    def counterfactual_reward(self, s, a, s_prime, z_values):
        """Compute average reward under different latent scenarios"""
        rewards = []
        for z in z_values:
            z_tensor = torch.tensor(z).repeat(len(s), 1).to(self.device)
            reward, _ = self.discriminator.f(s, a, s_prime, z_tensor)
            rewards.append(reward.mean().item())
        return rewards

    def save(self, path: str):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'prior': self.prior.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy': self.policy.state_dict(),
            'state_encoder': self.state_encoder.state_dict() if self.state_encoder else None
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.prior.load_state_dict(checkpoint['prior'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy.load_state_dict(checkpoint['policy'])
        if checkpoint['state_encoder'] and self.state_encoder:
            self.state_encoder.load_state_dict(checkpoint['state_encoder'])

    def update_policy(self, states, actions, next_states, rewards, old_log_probs):
        """PPO-style policy update with value network"""
        # Compute TD targets
        with torch.no_grad():
            next_values = self.value_net(next_states).squeeze()
        targets = rewards + self.gamma * next_values
        values = self.value_net(states).squeeze(-1)

        # Compute advantages
        advantages = targets - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get current policy log probs
        dist = self.policy(states)
        log_probs = dist.log_prob(actions.squeeze())

        # PPO loss
        ratios = torch.exp(log_probs - old_log_probs.detach())
        clipped_ratios = torch.clamp(ratios, 1-0.2, 1+0.2)
        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

        assert values.shape == targets.shape, f"[PPO] Shape mismatch: {values.shape} vs {targets.shape}"
        # Value loss
        value_loss = F.mse_loss(values, targets)

        # Entropy bonus
        entropy = dist.entropy().mean()

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self.optimizer_pi.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.optimizer_pi.step()

        return total_loss.item()

    def update_causal_discriminator(
        self,
        expert_data: Tuple[torch.Tensor],
        agent_data: Tuple[torch.Tensor],
        beta_in: float = 1.0,
        batch_size: int = 64,
        epochs: int = 3,
        num_z_samples: int = 5,
        current_epoch: int = 0,
        total_epochs: int = 100
    ) -> Dict[str, float]:
        # Unpack data
        s_e, a_e, s_prime_e = [x.to(self.device) for x in expert_data]
        s_pi, a_pi, s_prime_pi, log_pi_agent = [x.to(self.device) for x in agent_data]

        # Compute expert log_probs using current policy
        with torch.no_grad():
            dist_e = self.policy(s_e)
            log_pi_expert = dist_e.log_prob(a_e.squeeze()).unsqueeze(1)

        # Combine data
        all_s = torch.cat([s_e, s_pi])
        all_a = torch.cat([a_e, a_pi])
        all_s_prime = torch.cat([s_prime_e, s_prime_pi])
        all_log_pi = torch.cat([log_pi_expert, log_pi_agent])
        labels = torch.cat([
            torch.ones(len(s_e), 1, device=self.device),
            torch.zeros(len(s_pi), 1, device=self.device)
        ])

        dataset = TensorDataset(all_s, all_a, all_s_prime, all_log_pi, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_loss = 0.0
        total_inv_loss = 0.0
        total_kl = 0.0

        for epoch in range(epochs):
            ep_loss = 0.0
            ep_inv = 0.0
            ep_kl = 0.0
            ep_batches = 0

            for s, a, s_p, log_pi, y in loader:
                s = s.detach()
                a = a.detach()
                s_p = s_p.detach()
                log_pi = log_pi.detach()
                y = y.detach()

                batch_size = s.size(0)

                # Generate multiple z samples
                with torch.no_grad():
                    a_enc_a = self.action_encoder(a)
                    z_base, _, _ = self.encoder(s, a_enc_a)
                    z_samples = z_base.unsqueeze(0) + torch.randn(
                        num_z_samples, batch_size, self.encoder.latent_dim, device=self.device
                    )

                    # Compute invariance loss separately
                    inv_loss = self.discriminator.invariance_loss(s, a_enc_a, s_p, z_samples)
                    inv_loss_scalar = inv_loss.item()

                # Encode latent
                a_enc_kl = self.action_encoder(a)
                z_kl, mu, std = self.encoder(s, a_enc_kl)
                q_dist = Normal(mu, std)
                p_dist = self.prior(batch_size)
                kl = torch.clamp(kl_divergence(q_dist, p_dist).mean(), min=0, max=10)

                # print(f"[DEBUG] z_base mean: {z_base.mean().item():.4f}, std: {z_base.std().item():.4f}")
                # print(f"[DEBUG] z_samples std: {z_samples.std(dim=0).mean().item():.4f}")
                # print(f"[DEBUG] mu mean: {mu.mean().item():.4f}, std mean: {std.mean().item():.4f}")

                # Get discriminator outputs
                a_enc_disc = self.action_encoder(a)
                z_disc, _, _ = self.encoder(s, a_enc_disc)
                f_out, _ = self.discriminator.f(s, a_enc_disc, s_p, z_disc)
                d_pred = self.discriminator.D(f_out, log_pi)

                # Compute losses
                disc_loss = F.binary_cross_entropy(d_pred, y)

                # Combine losses
                annealed_penalty = self.discriminator.invariance_penalty * (current_epoch / total_epochs)
                mi_proxy = mu.pow(2).mean()
                beta = 0.1 + 0.9 * min(beta_in, current_epoch / (total_epochs / 2))
                loss = disc_loss + beta * kl + annealed_penalty * inv_loss_scalar - 1e-2 * mi_proxy # λ = 1e-2
                loss_scalar = loss.item()

                # print(f"[LOSS DEBUG] disc: {disc_loss.item():.4f} | kl: {kl.item():.4f} | inv: {inv_loss_scalar:.4f}")

                # Optimize
                self.optimizer_d.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
                self.optimizer_d.step()
                # NOTE : inv_loss is detached; no gradients flow through z_samples.
                # To optimize directly, remove .item() call and retain inv_loss in loss

                # Log intermediate variables
                ep_loss += loss.item()
                ep_inv += inv_loss_scalar
                ep_kl += kl.item()
                ep_batches += 1

                # Explicitly delete intermediate tensors
                del z_kl, mu, std, q_dist, p_dist, kl, z_disc, f_out, d_pred, disc_loss, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            if ep_batches > 0:
                self.logger.log({
                    "epoch_kl": ep_kl / ep_batches,
                    "epoch_inv_loss": ep_inv / ep_batches,
                    "epoch_beta": beta
                })

            total_loss += ep_loss
            total_inv_loss += ep_inv
            total_kl += ep_kl

        avg_loss = total_loss / len(loader)
        return {
            "total_loss": avg_loss,
            "invariance_loss": total_inv_loss / len(loader),
            "kl_divergence": total_kl / len(loader)
        }

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

        if hasattr(env, 'get_all_states'):
            print("[CausalAIRL] Using full GridWorld state enumeration.")
            raw_states = np.array(env.get_all_states(), dtype=np.float32)
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
                            step_out = env.step(a)
                            if len(step_out) == 5:
                                s, _, terminated, truncated, _ = step_out
                            else:
                                s, _, terminated, truncated = step_out

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

        states = self.state_encoder(torch.tensor(raw_states, dtype=torch.float32, device=self.device))
        # dummy_actions = torch.zeros(len(states), dtype=torch.long, device=self.device)
        dummy_actions = torch.arange(self.action_dim, device=self.device).repeat(len(states) // self.action_dim + 1)[:len(states)]
        actions = self.action_encoder(dummy_actions)
        assert actions.shape[1] == self.action_dim, \
            f"[CausalAIRL] Encoded actions shape mismatch: got {actions.shape[1]}, expected {self.action_dim}"

        z_tensor = torch.full((len(states), self.latent_dim), fill_value=0.0, device=self.device)
        z_tensor[:, 0] = float(z_value)

        with torch.no_grad():
            r_inv = self.discriminator.r_invariant(states)
            r_causal = self.discriminator.r_causal(torch.cat([states, actions, z_tensor], dim=-1))
            total = r_inv + r_causal

        return (
            total.cpu().numpy(),
            r_inv.cpu().numpy(),
            r_causal.cpu().numpy()
        )

    def train(
        self,
        cfg: dict,
        env: BaseEnv,
        demos: list
    ) -> Tuple[np.ndarray, dict, dict]:
        """Full training loop for Causal AIRL"""
        # Prepare expert data
        s_e, a_e, s_pe = self._prepare_expert_data(demos)
        
        # Training loop
        for it in range(cfg['irl']['max_iters']):
            # Collect agent rollouts
            agent_data = self._collect_agent_rollouts(
                env, cfg['train']['batch_size']
            )
            s_pi, a_pi, s_ppi, log_pi = self._process_agent_data(agent_data)
            
            # Update components
            d_metrics = self.update_causal_discriminator(
                (s_e, a_e, s_pe),
                (s_pi, a_pi, s_ppi, log_pi),
                beta_in=1.0,
                batch_size=cfg['train']['batch_size'],
                epochs=cfg['train']['epochs'],
                num_z_samples=cfg['irl'].get('num_z_samples', 10),
                current_epoch=it,
                total_epochs=cfg['irl']['max_iters']
            )
            self.logger.log(d_metrics)
            
            # Update policy with causal rewards
            with torch.no_grad():
                a_enc = self.action_encoder(a_pi)
                z, _, _ = self.encoder(s_pi, a_enc)
                a_enc = self.action_encoder(a_pi)
                rewards, _ = self.discriminator.f(s_pi, a_enc, s_ppi, z)
                rewards = rewards.squeeze(-1)
            
            policy_loss = self.update_policy(s_pi, a_pi, s_ppi, rewards, log_pi)
            self.logger.log({"policy_loss": policy_loss})

        # Extract reward components
        learned_reward, inv_reward, causal_reward = self.extract_reward_components(env)
        per_z_rewards = []
        if getattr(env, "confounder_values", None):
            for z in env.confounder_values:
                r_z, _, _ = self.extract_reward_components_for_z(env, z)
                per_z_rewards.append(r_z)


        if per_z_rewards:
            reward_var_z = np.var(
                np.stack([r.flatten() for r in per_z_rewards], axis=0),
                axis=0
            ).mean()
            reward_var_z = float(reward_var_z)
        else:
            reward_var_z = None  # or np.nan, depending on your visualisation expectations

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

    def extract_reward_components(self, env) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute reward components across representative states.
        Handles:
        - Custom GridWorld (via env.get_all_states)
        - Gymnasium CartPole-v1 (via dynamic rollouts)
        Returns: (total_reward, invariant_reward, causal_reward)
        """
        self.discriminator.eval()
        self.encoder.eval()

        if hasattr(env, 'get_all_states'):
            print("[CausalAIRL] Using full GridWorld state enumeration.")
            raw_states = np.array(env.get_all_states(), dtype=np.float32)
        else:
            print("[CausalAIRL] Using replay-based sampling for reward extraction.")

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
                            step_out = env.step(a)
                            if len(step_out) == 5:
                                s, _, terminated, truncated, _ = step_out
                            else:
                                s, _, terminated, truncated = step_out

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

        states = self.state_encoder(torch.tensor(raw_states, dtype=torch.float32, device=self.device))
        # dummy_actions = torch.zeros(len(states), dtype=torch.long, device=self.device)
        dummy_actions = torch.arange(self.action_dim, device=self.device).repeat(len(states) // self.action_dim + 1)[:len(states)]
        actions = self.action_encoder(dummy_actions)
        assert actions.shape[1] == self.action_dim, \
            f"[CausalAIRL] Encoded actions shape mismatch: got {actions.shape[1]}, expected {self.action_dim}"

        with torch.no_grad():
            z, _, _ = self.encoder(states, actions)
            total_reward, inv_reward, causal_reward = self.get_reward_components(states, actions, states)

        return (
            total_reward.cpu().numpy(),
            inv_reward.cpu().numpy(),
            causal_reward.cpu().numpy()
        )

    def _prepare_expert_data(self, demos):
        """Convert expert demos to tensors"""
        s_list, a_list, s_prime_list = [], [], []
        for traj in demos:
            for s, a, _, s_prime in traj:
                s_tensor = self.state_encoder(torch.FloatTensor(s).unsqueeze(0)).squeeze(0)
                s_prime_tensor = self.state_encoder(torch.FloatTensor(s_prime).unsqueeze(0)).squeeze(0)
                s_list.append(s_tensor)
                a_list.append(torch.LongTensor([a]))
                s_prime_list.append(s_prime_tensor)
        return (
            torch.stack(s_list).to(self.device),
            torch.stack(a_list).to(self.device),
            torch.stack(s_prime_list).to(self.device)
        )

    def _collect_agent_rollouts(self, env, episodes=10):
        """Collect policy rollouts for training"""
        data = []
        self.policy.eval()

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
                    s_raw = torch.FloatTensor(s).unsqueeze(0).to(self.device)
                    s_tensor = self.state_encoder(s_raw)
                    dist = self.policy(s_tensor)
                    a = dist.sample().detach()
                    logp = dist.log_prob(a)

                    # Handle discrete and continuous actions
                    if isinstance(a, torch.Tensor):
                        if a.ndim == 0 or (a.ndim == 1 and a.shape[0] == 1):
                            action_to_env = a.item()
                        else:
                            action_to_env = a.cpu().numpy()
                    else:
                        action_to_env = a

                    s_next, _, done, _ = env.step(action_to_env)
                    s_next_raw = torch.FloatTensor(s_next).unsqueeze(0).to(self.device)
                    s_next_tensor = self.state_encoder(s_next_raw)
                    data.append((s_tensor.squeeze(0), a, s_next_tensor.squeeze(0), logp))
                    s = s_next
                    steps += 1
        else:
            # CartPole or other continuous environment logic
            if hasattr(env, "spec") and env.spec is not None:
                max_steps = env.spec.max_episode_steps
            else:
                max_steps = 200
            for _ in range(episodes):
                s = env.reset()
                done = False
                steps = 0
                while not done and steps < max_steps:
                    s_tensor = self.state_encoder(torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0))
                    dist = self.policy(s_tensor)
                    a = dist.sample().detach()
                    logp = dist.log_prob(a)

                    if isinstance(a, torch.Tensor):
                        if a.ndim == 0 or (a.ndim == 1 and a.shape[0] == 1):
                            action_to_env = a.item()
                        else:
                            action_to_env = a.cpu().numpy()
                    else:
                        action_to_env = a

                    s_next, _, done, _ = env.step(action_to_env)
                    s_next_tensor = self.state_encoder(torch.tensor(s_next, dtype=torch.float32, device=self.device).unsqueeze(0))
                    data.append((s_tensor.squeeze(0), a, s_next_tensor.squeeze(0), logp))
                    s = s_next
                    steps += 1

        return data

    def _process_agent_data(self, agent_data):
        """Convert rollout data to tensors"""
        s, a, s_prime, logp = map(torch.stack, zip(*agent_data))
        return s, a, s_prime, logp
