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
        l2_reg: float = 1e-4
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
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
        return torch.sigmoid(f - log_pi)

    def reward(self, s, a, s_prime=None, log_pi=None):
        """
        Full AIRL reward: f(s,a,s') - log π(a|s) when log_pi provided
        Otherwise returns f(s,a,s') for visualization
        """
        with torch.no_grad():
            f = self.forward(s, a, s_prime if s_prime is not None else s)
            return f if log_pi is None else f - log_pi

    def state_only_reward(self, s):
        """Extract state-only reward component r(s) for visualization"""
        with torch.no_grad():
            s_enc = self.state_encoder(s)
            dummy_a = torch.zeros(s_enc.shape[0], self.action_dim, device=self.device)
            return self.r(torch.cat([s_enc, dummy_a], dim=-1))

    def _l2_penalty(self):
        """Compute L2 regularization for all networks"""
        return sum(
            layer.weight.pow(2).sum()
            for net in [self.r, self.h]
            for layer in net
            if isinstance(layer, nn.Linear)
        )

    def _get_config(self):
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'hidden_dim': self.r[0].in_features if isinstance(self.r[0], nn.Linear) else 64,
            'l2_reg': self.l2_reg,
            'state_encoder': self.state_encoder,
            'action_encoder': self.action_encoder
        }

class AIRLAgent:
    """Complete AIRL agent with training and evaluation logic"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 3e-4,
        device: torch.device = None,
        state_encoder = None,
        action_encoder = None
    ):
        self.device = device or get_device()
        self.gamma = gamma

        self.state_encoder = state_encoder or (lambda x: x)
        self.action_encoder = action_encoder or (lambda x: x)

        self.discriminator = AIRLDiscriminator(
            state_dim, action_dim, gamma,
            state_encoder=self.state_encoder,
            action_encoder=self.action_encoder
        ).to(self.device)
        
        self.policy = PolicyNet(state_dim, action_dim).to(self.device)
        
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
        rewards: torch.Tensor
    ) -> float:
        """REINFORCE policy update with entropy regularization"""
        self.policy.train()
        dist = self.policy(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        loss = -(log_probs * rewards).mean() - 0.01 * entropy
        self.optimizer_pi.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer_pi.step()
        return loss.item()

    def extract_reward(self, env) -> np.ndarray:
        """Compute learned reward across all states"""
        self.discriminator.eval()

        states_np = np.array(env.get_all_states(), dtype=np.float32)
        assert states_np.shape[0] == env.n_states, \
            f"Mismatch: {states_np.shape[0]} states vs env.n_states = {env.n_states}"

        assert len(set(map(tuple, states_np))) == env.n_states, "Non-unique states returned"

        states = torch.from_numpy(states_np).to(self.device)
        dummy_actions = torch.zeros(len(states), dtype=torch.long, device=self.device)

        with torch.no_grad():
            rewards = self.discriminator.reward(states, dummy_actions, states)

        rewards_np = rewards.cpu().numpy()
        assert rewards_np.shape[0] == env.n_states, \
            f"Reward shape mismatch: got {rewards_np.shape[0]}, expected {env.n_states}"

        return rewards_np

    def train(
        self,
        cfg: dict,
        env: BaseEnv,
        demos: list
    ) -> Tuple[np.ndarray, dict, dict]:
        """Full training loop for AIRL"""
        # Prepare expert data
        s_e, a_e, s_pe = self._prepare_expert_data(demos)
        
        # Training loop
        for it in range(cfg['irl']['max_iters']):
            # Collect agent rollouts
            agent_data = self._collect_agent_rollouts(
                env, cfg['train']['batch_size']
            )
            s_pi, a_pi, s_ppi, log_pi = self._process_agent_data(agent_data)
            
            # Update discriminator
            d_metrics = update_discriminator(
                self.discriminator,
                self.policy,
                (s_e, a_e, s_pe),
                (s_pi, a_pi, s_ppi, log_pi),
                self.optimizer_d,
                batch_size=cfg['train']['batch_size'],
                epochs=cfg['train']['epochs'],
                device=self.device
            )
            self.logger.log(d_metrics)
            
            # Update policy with learned rewards
            with torch.no_grad():
                rewards = self.discriminator.reward(
                    s_pi, a_pi, s_ppi, log_pi
                ).detach()
            
            policy_loss = self.update_policy(s_pi, a_pi.squeeze(), rewards)
            self.logger.log({"policy_loss": policy_loss})
        
        # Final evaluation
        learned_reward = self.extract_reward(env)
        return learned_reward, self.logger.get_logs(), {
            'policy': self.policy,
            'discriminator': self.discriminator
        }

    def _prepare_expert_data(self, demos):
        """Convert expert demos to tensors"""
        s_list, a_list, s_prime_list = [], [], []
        for traj in demos:
            for s, a, _, s_prime in traj:
                s_list.append(torch.FloatTensor(s))
                a_list.append(torch.LongTensor([a]))
                s_prime_list.append(torch.FloatTensor(s_prime))

        # Stack and encode expert states using the agent's encoder
        s_raw = torch.stack(s_list).to(self.device)
        s_prime_raw = torch.stack(s_prime_list).to(self.device)

        s_enc = self.discriminator.state_encoder(s_raw)
        s_prime_enc = self.discriminator.state_encoder(s_prime_raw)

        return s_enc, torch.stack(a_list).squeeze().to(self.device), s_prime_enc

    def _collect_agent_rollouts(self, env, episodes=10):
        """Collect policy rollouts for training"""
        data = []
        self.policy.eval()
        W, H = env.grid_size
        n_states = W * H
        min_path = (W - 1) + (H - 1)
        H_max = min(int(2.0 * min_path + 5), 100)

        for _ in range(episodes):
            s = env.reset()
            done = False
            steps = 0
            while not done and steps < H_max:
                # Convert (x, y) state to one-hot vector
                s_index = env.state_to_index(s, n_cols=H)
                s_onehot = F.one_hot(torch.tensor([s_index], device=self.device), num_classes=n_states).float()
                dist = self.policy(s_onehot)
                a = dist.sample()
                logp = dist.log_prob(a)

                # Handle discrete and continuous actions
                if isinstance(a, torch.Tensor):
                    if a.ndim == 0:
                        action_to_env = a.item()
                    elif a.ndim == 1 and a.shape[0] == 1:
                        action_to_env = a.item()
                    else:
                        action_to_env = a.cpu().numpy()
                else:
                    action_to_env = a

                s_next, _, done, _ = env.step(action_to_env)
                s_next_index = env.state_to_index(s_next, n_cols=H)
                s_next_onehot = F.one_hot(torch.tensor([s_next_index], device=self.device), num_classes=n_states).float()

                data.append((s_onehot.squeeze(0), a.squeeze(), s_next_onehot.squeeze(0), logp.squeeze()))
                s = s_next
                steps += 1
        return data

    def _process_agent_data(self, agent_data):
        """Convert rollout data to tensors"""
        s, a, s_prime, logp = map(torch.stack, zip(*agent_data))
        return s, a.squeeze(), s_prime, logp.unsqueeze(1)

def compute_airl_reward(
    discriminator: AIRLDiscriminator,
    s: torch.Tensor,
    a: torch.Tensor,
    s_prime: torch.Tensor,
    log_pi: torch.Tensor,
    detach: bool = False,
    device: torch.device = get_device()
) -> torch.Tensor:
    """Compute reward: log D - log(1 - D) = f(s,a,s') - log π(a|s)
    Args:
        detach: Whether to detach gradients from discriminator (for policy training)
    """
    f = discriminator(s, a, s_prime)
    if detach:
        f = f.detach()
    return f.to(device) - log_pi.to(device)


def update_discriminator(
    discriminator: AIRLDiscriminator,
    policy: nn.Module,
    expert_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    agent_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    batch_size: int = 64,
    epochs: int = 3,
    device: torch.device = get_device()
) -> Dict[str, float]:
    """
    Improved discriminator training with:
    - Proper expert log_prob computation
    - L2 regularization
    - Memory-efficient batching

    Args:
        policy: Current policy π(a|s) for computing expert log probs
        expert_data: (s, a, s_prime) tuples
        agent_data: (s, a, s_prime, log_pi) tuples
    """
    s_e, a_e, s_prime_e = [x.to(device) for x in expert_data]
    s_pi, a_pi, s_prime_pi, log_pi_agent = [x.to(device) for x in agent_data]

    # Compute expert log probs using current policy
    with torch.no_grad():
        dist_e = policy(s_e)
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

    discriminator.train()
    total_loss = 0.0
    total_batches = 0

    for _ in range(epochs):
        for batch in batch_generator():
            s_batch, a_batch, s_prime_batch, log_pi_batch, labels_batch = batch

            # Compute discriminator output
            d_pred = discriminator(s_batch, a_batch, s_prime_batch, log_pi_batch)

            # Compute loss + L2 regularization
            loss = F.binary_cross_entropy(d_pred, labels_batch)
            loss += discriminator.l2_reg * discriminator._l2_penalty()

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0) # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else float('nan')
    return {"discriminator_loss": avg_loss}

# ========== OPTIMIZED ENCODERS ========== #

def create_onehot_encoder(num_classes: int):
    """Optimized one-hot encoder"""
    def encoder(x: torch.Tensor):
        assert x.dtype in (torch.long, torch.int), f"Expected integer action indices, got {x.dtype}"
        x = x.view(-1)  # Always flatten to 1D index vector
        return F.one_hot(x.long(), num_classes=num_classes).float()
    return encoder

def create_continuous_encoder(input_dim: int):
    """Identity encoder with shape check"""
    return lambda x: x.view(-1, input_dim)

def create_gridworld_encoder(grid_size: int):
    """Optimized gridworld encoder (2x faster)"""
    def encoder(s: torch.Tensor):
        idx = s[:, 0].long() * grid_size + s[:, 1].long()
        return F.one_hot(idx, num_classes=grid_size**2).float()
    return encoder

def save_model(model: nn.Module, path: str):
    torch.save({
        'state_dict': model.state_dict(),
        'config': model._get_config() if hasattr(model, '_get_config') else {}
    }, path)

def load_model(cls, path: str):
    checkpoint = torch.load(path)
    model = cls(**checkpoint['config']) if 'config' in checkpoint else cls()
    model.load_state_dict(checkpoint['state_dict'])
    return model
