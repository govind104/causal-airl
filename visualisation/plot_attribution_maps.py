import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import json

from irl.airl import AIRLDiscriminator, create_gridworld_encoder, create_cartpole_encoder
from irl.causal_airl import CausalDiscriminator


def load_states(states_path):
    states = np.load(states_path)
    return torch.tensor(states, dtype=torch.float32)

def load_actions(actions_path):
    actions = np.load(actions_path)
    return torch.tensor(actions, dtype=torch.long)

def load_env_shape(env_data_path):
    with open(env_data_path, "r") as f:
        env_data = json.load(f)
    if "grid_size" in env_data:
        return tuple(env_data["grid_size"])
    else:
        raise ValueError("grid_size not found in env_data.json")


def load_terminal_states(env_data_path):
    with open(env_data_path, "r") as f:
        env_data = json.load(f)
    return env_data.get("terminal_states", [])


def compute_saliency(discriminator, states, device, actions=None, action_encoder=None):
    states = states.clone().detach().to(device).requires_grad_(True)
    encoded_s = discriminator.state_encoder(states)

    if actions is not None and action_encoder is not None:
        actions = actions.to(device)
        encoded_a = action_encoder(actions)
        sa = torch.cat([encoded_s, encoded_a], dim=1)
    else:
        sa = encoded_s

    with torch.enable_grad():
        rewards = discriminator.compute_reward(sa).sum()
        rewards.backward()
        saliency = states.grad.abs().sum(dim=1).detach().cpu().numpy()
    return saliency

def plot_saliency_map(saliency, grid_shape, terminals, title, save_path):
    heatmap = saliency.reshape(grid_shape)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(heatmap, cmap="hot", origin="upper")
    plt.colorbar(im, label="Saliency (|∇R|)")
    plt.title(title)
    for (i, j) in terminals:
        plt.text(j, i, '★', ha='center', va='center', color='white', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[Saved] {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot reward attribution map using gradients ∇R(s)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to discriminator.pt (AIRL or Causal-AIRL)")
    parser.add_argument("--states_path", type=str, required=True,
                        help="Path to states.npy file (NxD tensor)")
    parser.add_argument("--actions_path", type=str, default=None,
                        help="Optional path to actions.npy if reward is R(s,a)")
    parser.add_argument("--env_data", type=str, required=True,
                        help="Path to env_data.json (for grid info)")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Output path to save heatmap")
    parser.add_argument("--title", type=str, default="Reward Attribution")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to config.json (for action_dim)")

    args = parser.parse_args()
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    # Load states
    states = load_states(args.states_path)
    actions = load_actions(args.actions_path) if args.actions_path and os.path.exists(args.actions_path) else None

    # Rebuild encoder based on config.json
    with open(args.config_path, "r") as f:
        cfg = json.load(f)

    state_encoding = cfg.get("state_encoding", "raw")
    action_dim = cfg.get("action_dim", 2)
    action_encoding = cfg.get("action_encoding", "onehot")

    method = cfg["irl"]["method"]
    if state_encoding == "raw":
        state_encoder = create_cartpole_encoder()
    elif state_encoding == "onehot":
        state_encoder = create_gridworld_encoder()
    else:
        raise ValueError(f"Unsupported state_encoding: {state_encoding}")

    action_encoder = None
    if actions is not None:
        if action_encoding == "onehot":
            def action_encoder(a):
                return F.one_hot(a.view(-1), num_classes=action_dim).float()
        else:
            raise ValueError(f"Unsupported action_encoding: {action_encoding}")

    # Construct discriminator
    if method == "airl":
        discriminator = AIRLDiscriminator(
            state_dim=states.shape[1],
            action_dim=action_dim,
            state_encoder=state_encoder
        )
    else:
        discriminator = CausalDiscriminator(
            state_dim=states.shape[1],
            action_dim=action_dim,
            latent_dim=cfg.get("latent_dim", 2),
            gamma=cfg.get("gamma", 0.99),
            invariance_penalty=cfg.get("invariance_penalty", 0.1)
        )
        discriminator.state_encoder = state_encoder  # required for saliency
        if action_encoder:
            discriminator.action_encoder = action_encoder
    discriminator.load_state_dict(torch.load(args.model_path, map_location=device))
    discriminator = discriminator.to(device).eval()

    # Compute saliency
    saliency = compute_saliency(discriminator, states, device, actions, action_encoder)

    # Plot
    shape = load_env_shape(args.env_data)
    terminals = load_terminal_states(args.env_data)
    plot_saliency_map(saliency, shape, terminals, args.title, args.save_path)
