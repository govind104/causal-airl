import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional, Callable

from envs.environments import BaseEnv
from experiments.logger import TrainingLogger
from models.policy import PolicyNet


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CausalPrior(nn.Module):
    """Learnable prior with causal structure"""
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, batch_size):
        return Normal(torch.zeros(batch_size, self.latent_dim),
                     torch.ones(batch_size, self.latent_dim))

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

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        hidden = self.net(x)
        mu = self.mu_net(hidden)
        logvar = self.logvar_net(hidden)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, std

class CausalDiscriminator(nn.Module):
    """Causal-AIRL discriminator with invariance constraints"""
    def __init__(self, state_dim, action_dim, latent_dim, gamma=0.99, invariance_penalty=0.1):
        super().__init__()
        self.gamma = gamma
        self.invariance_penalty = invariance_penalty

        # State encoder (invariant pathway)
        self.s_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # State-action encoder (causal pathway)
        self.sa_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # Reward networks
        self.r_invariant = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.r_causal = nn.Sequential(
            nn.Linear(64 + latent_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Shaping function (causal)
        self.h = nn.Sequential(
            nn.Linear(64 + latent_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def f(self, s, a, s_prime, z):
        """Causal reward decomposition: f = r_causal + γh(s') - h(s). Returns (total_reward, invariant_reward)        """
        s_enc = self.s_encoder(s)
        sa_enc = self.sa_encoder(torch.cat([s, a], dim=-1))

        # Invariant reward (z-independent)
        r_inv = self.r_invariant(s_enc)

        # Causal reward (z-dependent)
        r_causal = self.r_causal(torch.cat([sa_enc, z], dim=-1))

        # Full reward (invariant + causal)
        r_full = r_inv + r_causal

        # Potential-based shaping
        h_s = self.h(torch.cat([self.s_encoder(s), z], dim=-1))
        h_sp = self.h(torch.cat([self.s_encoder(s_prime), z], dim=-1))

        return (r_full + self.gamma * h_sp - h_s), r_inv

    def D(self, f, log_pi):
        return torch.sigmoid(f - log_pi)

    def invariance_loss(self, s, z_samples):
        """
        Compute reward variance across different latent confounders
        Args:
            s: states [batch_size, state_dim]
            z_samples: latent samples [num_samples, batch_size, z_dim]
        """
        batch_size = s.size(0)
        num_samples = z_samples.size(0)

        # Encode states once [batch_size, hidden]
        s_enc = self.s_encoder(s)

        # Expand to [num_samples, batch_size, hidden]
        s_enc_expanded = s_enc.unsqueeze(0).expand(num_samples, -1, -1)

        # Compute invariant rewards [num_samples, batch_size, 1]
        rewards = self.r_invariant(s_enc_expanded)

        # Compute variance across samples (per state)
        return rewards.var(dim=0).mean()  # [batch_size, 1] -> scalar

def update_causal_discriminator(
    discriminator: CausalDiscriminator,
    encoder: CausalEncoder,
    prior: CausalPrior,
    policy: nn.Module,
    expert_data: Tuple[torch.Tensor],
    agent_data: Tuple[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    beta: float = 1.0,
    batch_size: int = 64,
    epochs: int = 3,
    num_z_samples: int = 5,
    device: torch.device = get_device(),
    current_epoch: int = 0,
    total_epochs: int = 100
) -> Dict[str, float]:
    # Unpack data
    s_e, a_e, s_prime_e = [x.to(device) for x in expert_data]
    s_pi, a_pi, s_prime_pi, log_pi_agent = [x.to(device) for x in agent_data]

    # Compute expert log_probs using current policy
    with torch.no_grad():
        log_pi_expert = policy.log_prob(s_e, a_e).unsqueeze(1)

    # Combine data
    all_s = torch.cat([s_e, s_pi])
    all_a = torch.cat([a_e, a_pi])
    all_s_prime = torch.cat([s_prime_e, s_prime_pi])
    all_log_pi = torch.cat([log_pi_expert, log_pi_agent])
    labels = torch.cat([
        torch.ones(len(s_e), 1, device=device),
        torch.zeros(len(s_pi), 1, device=device)
    ])

    dataset = TensorDataset(all_s, all_a, all_s_prime, all_log_pi, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    total_inv_loss = 0.0
    total_kl = 0.0

    for _ in range(epochs):
        for s, a, s_p, log_pi, y in loader:
            batch_size = s.size(0)

            # Generate multiple z samples
            with torch.no_grad():
                z_base, _, _ = encoder(s, a)
                z_samples = z_base.unsqueeze(0) + torch.randn(
                    num_z_samples, batch_size, encoder.latent_dim, device=device
                )

            # Encode latent
            z, mu, std = encoder(s, a)
            q_dist = Normal(mu, std)
            p_dist = prior(batch_size)
            kl = torch.clamp(kl_divergence(q_dist, p_dist).mean(), min=0, max=10)

            # Get discriminator outputs
            f_out, _ = discriminator.f(s, a, s_p, z)
            d_pred = discriminator.D(f_out, log_pi)

            # Compute losses
            disc_loss = F.binary_cross_entropy(d_pred, y)
            inv_loss = discriminator.invariance_loss(s, z_samples)

            # Combine losses
            annealed_penalty = discriminator.invariance_penalty * (current_epoch / total_epochs) # Anneal invariance penalty
            loss = disc_loss + beta * kl + annealed_penalty * inv_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            total_inv_loss += inv_loss.item()
            total_kl += kl.item()

    avg_loss = total_loss / len(loader)
    return {
        "total_loss": avg_loss,
        "invariance_loss": total_inv_loss / len(loader),
        "kl_divergence": total_kl / len(loader)
    }

class CausalAIRL(nn.Module):
    """End-to-end Causal-AIRL agent with proper causal structure"""
    def __init__(self, state_dim, action_dim, latent_dim=2, gamma=0.99,
                 lr=3e-4, invariance_penalty=0.1, device=None):
        super().__init__()
        self.device = get_device() if device is None else device
        self.latent_dim = latent_dim
        self.gamma = gamma

        # Components
        self.encoder = CausalEncoder(state_dim, action_dim, latent_dim).to(self.device)
        self.prior = CausalPrior(latent_dim).to(self.device)
        self.discriminator = CausalDiscriminator(
            state_dim, action_dim, latent_dim, gamma, invariance_penalty
        ).to(self.device)

        # Policy and value network
        self.policy = None
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        # Optimizers
        self.optim_d = torch.optim.Adam(
            list(self.discriminator.parameters()) +
            list(self.encoder.parameters()),
            lr=lr
        )
        self.optim_pi = None

    def set_policy(self, policy):
        self.policy = policy.to(self.device)
        self.optim_pi = torch.optim.Adam(
            list(self.policy.parameters()) +
            list(self.value_net.parameters()),
            lr=3e-4
        )

    def train_step(self, expert_batch, agent_batch):
        # Update discriminator
        d_metrics = update_causal_discriminator(
            self.discriminator,
            self.encoder,
            self.prior,
            self.policy,
            expert_batch,
            agent_batch,
            self.optim_d,
            device=self.device
        )

        # Update policy with causal rewards
        s_pi, a_pi, s_prime_pi, _ = agent_batch

        with torch.no_grad():
            # Sample latent for policy update
            z, _, _ = self.encoder(s_pi, a_pi)
            rewards, _ = self.discriminator.f(s_pi, a_pi, s_prime_pi, z)

        # Policy update (PPO-style for stability)
        policy_loss = self.update_policy(s_pi, a_pi, s_prime_pi, rewards)
        return {**d_metrics, "policy_loss": policy_loss}

    def update_policy(self, states, actions, next_states, rewards):
        # Compute TD targets
        with torch.no_grad():
            next_values = self.value_net(next_states).squeeze()
        targets = rewards + self.gamma * next_values
        values = self.value_net(states).squeeze()

        # Compute advantages
        advantages = targets - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get policy log probs
        log_probs = self.policy.log_prob(states, actions)

        # PPO loss
        old_log_probs = log_probs.detach()
        ratios = torch.exp(log_probs - old_log_probs)

        # Clipped policy loss
        clipped_ratios = torch.clamp(ratios, 1-0.2, 1+0.2)
        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

        # Value loss
        value_loss = F.mse_loss(values, targets)

        # Entropy bonus
        entropy = self.policy.entropy().mean()

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self.optim_pi.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.optim_pi.step()

        return total_loss.item()

    def compute_returns(self, rewards):
        """Compute discounted returns (simplified for batch)"""
        returns = torch.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R
            returns[t] = R
        return (returns - returns.mean()) / (returns.std() + 1e-8)

    def get_reward(self, s, a, s_prime, z=None):
        """Get causal-aware reward"""
        with torch.no_grad():
            if z is None:
                # Sample latent if not provided
                z, _, _ = self.encoder(s, a)
            reward, _ = self.discriminator.f(s, a, s_prime, z)
        return reward

    def get_invariant_reward(self, s):
        """Get Z-invariant reward component"""
        with torch.no_grad():
            s_enc = self.discriminator.s_encoder(s)
            reward = self.discriminator.r_invariant(s_enc)
        return reward

    def get_reward_components(self, s, a, s_prime):
        """Returns (total_reward, invariant_component, causal_component)"""
        with torch.no_grad():
            z, _, _ = self.encoder(s, a)
            s_enc = self.discriminator.s_encoder(s)
            sa_enc = self.discriminator.sa_encoder(torch.cat([s, a], -1))
            
            r_inv = self.discriminator.r_invariant(s_enc)
            r_causal = self.discriminator.r_causal(torch.cat([sa_enc, z], -1))
            
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
            'policy': self.policy.state_dict() if self.policy else None
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.prior.load_state_dict(checkpoint['prior'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        if self.policy and checkpoint['policy']:
            self.policy.load_state_dict(checkpoint['policy'])

class CausalAIRLAgent:
    """Complete Causal AIRL agent with training and evaluation"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 2,
        gamma: float = 0.99,
        invariance_penalty: float = 0.1,
        lr: float = 3e-4,
        device: torch.device = None
    ):
        self.device = device or get_device()
        self.latent_dim = latent_dim
        self.gamma = gamma
        
        self.encoder = CausalEncoder(state_dim, action_dim, latent_dim).to(self.device)
        self.prior = CausalPrior(latent_dim).to(self.device)
        self.discriminator = CausalDiscriminator(
            state_dim, action_dim, latent_dim, gamma, invariance_penalty
        ).to(self.device)
        self.policy = PolicyNet(state_dim, action_dim).to(self.device)
        
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
        
    def extract_reward_components_for_z(self, env, z_value):
        """Compute reward components across all states for a fixed latent z"""
        self.discriminator.eval()
        self.encoder.eval()
        
        states = torch.FloatTensor(env.get_all_states()).to(self.device)
        dummy_actions = torch.zeros(len(states), dtype=torch.long).to(self.device)
        
        # Create z tensor with batch dimension
        z = torch.tensor(z_value, dtype=torch.float, device=self.device).view(1, -1)
        z = z.repeat(len(states), 1)
        
        with torch.no_grad():
            s_enc = self.discriminator.s_encoder(states)
            sa_enc = self.discriminator.sa_encoder(torch.cat([states, dummy_actions], dim=-1))
            r_inv = self.discriminator.r_invariant(s_enc)
            r_causal = self.discriminator.r_causal(torch.cat([sa_enc, z], dim=-1))
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
            d_metrics = update_causal_discriminator(
                self.discriminator,
                self.encoder,
                self.prior,
                self.policy,
                (s_e, a_e, s_pe),
                (s_pi, a_pi, s_ppi, log_pi),
                self.optimizer_d,
                beta=1.0,
                batch_size=cfg['train']['batch_size'],
                epochs=cfg['train']['epochs'],
                current_epoch=it,
                total_epochs=cfg['irl']['max_iters'],
                device=self.device
            )
            self.logger.log(d_metrics)
            
            # Update policy with causal rewards
            with torch.no_grad():
                z, _, _ = self.encoder(s_pi, a_pi)
                rewards, _ = self.discriminator.f(s_pi, a_pi, s_ppi, z)
            
            policy_loss = self.update_policy(s_pi, a_pi, rewards)
            self.logger.log({"policy_loss": policy_loss})

        # Extract reward components
        learned_reward, inv_reward, causal_reward = self.extract_reward_components(env)
        per_z_rewards = []
        if getattr(env, "confounder_values", None):
            for z in env.confounder_values:
                r_z, _, _ = self.extract_reward_components_for_z(env, z)
                per_z_rewards.append(r_z)
        return learned_reward, self.logger.get_logs(), {
            'policy': self.policy,
            'discriminator': self.discriminator,
            'encoder': self.encoder,
            'invariant_reward': inv_reward,
            'causal_reward': causal_reward,
            "per_z_rewards": per_z_rewards
        }

    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor
    ) -> float:
        """REINFORCE policy update with entropy regularization"""
        self.policy.train()
        dist = self.policy(states)
        log_probs = dist.log_prob(actions.squeeze())
        entropy = dist.entropy().mean()
        
        loss = -(log_probs * rewards.squeeze()).mean() - 0.01 * entropy
        self.optimizer_pi.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer_pi.step()
        return loss.item()

    def extract_reward_components(self, env) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute reward components across all states"""
        if not hasattr(env, "get_all_states"):
            print("[Warning] extract_reward_components called on continuous env without get_all_states(). Skipping.")
            return None, None, None
        self.discriminator.eval()
        self.encoder.eval()
        
        states = torch.FloatTensor(env.get_all_states()).to(self.device)
        dummy_actions = torch.zeros(len(states), dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            z, _, _ = self.encoder(states, dummy_actions)
            total_reward, inv_reward, causal_reward = self.discriminator.get_reward_components(
                states, dummy_actions, states
            )
        
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
                s_list.append(torch.FloatTensor(s))
                a_list.append(torch.LongTensor([a]))
                s_prime_list.append(torch.FloatTensor(s_prime))
        return (
            torch.stack(s_list).to(self.device),
            torch.stack(a_list).to(self.device),
            torch.stack(s_prime_list).to(self.device)
        )

    def _collect_agent_rollouts(self, env, episodes=10):
        """Collect policy rollouts for training"""
        data = []
        self.policy.eval()
        for _ in range(episodes):
            s = env.reset()
            done = False
            while not done:
                s_tensor = torch.FloatTensor(s).to(self.device)
                dist = self.policy(s_tensor)
                a = dist.sample()
                logp = dist.log_prob(a)
                s_next, _, done, _ = env.step(a.cpu().numpy())
                s_next_tensor = torch.FloatTensor(s_next).to(self.device)
                data.append((s_tensor, a, s_next_tensor, logp))
                s = s_next
        return data

    def _process_agent_data(self, agent_data):
        """Convert rollout data to tensors"""
        s, a, s_prime, logp = map(torch.stack, zip(*agent_data))
        return s, a, s_prime, logp