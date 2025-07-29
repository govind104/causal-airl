import numpy as np
import torch

from envs.confounded_gridworld import ConfoundedGridWorld
from envs.utils import build_transition_matrix
from models.reward_network import RewardNet
from models.latent_encoder import LatentEncoder
from models.policy import PolicyNet
from irl.causal_airl import CausalAIRL
from experiments.eval import reward_correlation


def test_causal_airl_generalises_to_unseen_z(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = ConfoundedGridWorld(grid_size=(5, 5), terminal_states=[(4, 4)], gamma=0.99)
    all_demos = env.sample_expert_trajectories(n_trajectories=50, return_z=True)

    # Split: 40 demos for training (z=0), 10 held out (z=1)
    train_demos = [(z, traj) for z, traj in all_demos if z == 0][:40]
    heldout_demos = [(z, traj) for z, traj in all_demos if z == 1][:10]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reward_net = RewardNet(input_dim=31).to(device)
    encoder = LatentEncoder(state_dim=25, action_dim=4).to(device)
    policy = PolicyNet(state_dim=25, action_dim=4).to(device)

    airl = CausalAIRL(
        reward_net=reward_net,
        encoder=encoder,
        policy=policy,
        device=device,
        gamma=0.99,
        z_dim=2
    )

    airl.train(train_demos, env, epochs=5)

    # Evaluate generalisation to z=1 demos
    reward_map = airl.get_reward_map(env)
    true_reward = env.get_ground_truth_reward()
    corr = reward_correlation(true_reward, reward_map)

    print(f"[Causal-AIRL] Generalisation reward corr: {corr:.3f}")
    assert corr > 0.5, "Failed to generalise to unseen z=1 states"


if __name__ == "__main__":
    test_causal_airl_generalises_to_unseen_z()
    print("[PASS] test_causal_airl_generalises_to_unseen_z")
