import torch
import torch.nn as nn
import torch.distributions as D


class PolicyNet(nn.Module):
    """
    Stochastic policy π(a | s) for discrete action spaces.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        # Ensure input is properly shaped and float
        if state.dtype != torch.float32:
            state = state.float()
        logits = self.net(state)
        dist = D.Categorical(logits=logits)
        return dist

    def act(self, state):
        with torch.no_grad():
            dist = self.forward(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob
