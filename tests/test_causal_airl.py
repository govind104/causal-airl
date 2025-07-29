import numpy as np
import torch

from envs.confounded_gridworld import ConfoundedGridWorld
from envs.utils import build_transition_matrix, greedy_policy_from_map
from models.reward_network import RewardNet
from models.latent_encoder import LatentEncoder
from models.policy import PolicyNet
from irl.causal_airl import CausalAIRL
from experiments.eval import reward_correlation, policy_agreement


def test_causal_airl_reward_alignment(seed=123):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = ConfoundedGridWorld(grid_size=(5, 5), terminal_states=[(4, 4)], gamma=0.99)
    demos = env.sample_expert_trajectories(n_trajectories=30, optimality=1.0, return_z=True)

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

    airl.train(demos, env, epochs=5)

    reward_map = airl.get_reward_map(env)
    true_reward = env.get_ground_truth_reward()
    pi_learned = greedy_policy_from_map(reward_map, env)
    pi_expert = env.get_optimal_policy().reshape((env.n_rows, env.n_cols))

    corr = reward_correlation(true_reward, reward_map)
    agree = policy_agreement(pi_expert, pi_learned)

    print(f"[Causal-AIRL] Reward corr: {corr:.3f}, Policy agreement: {agree:.3f}")
    assert corr > 0.6, "Reward does not match ground truth well"
    assert agree > 0.7, "Policy does not match expert behavior"


if __name__ == "__main__":
    test_causal_airl_reward_alignment()
    print("[PASS] test_causal_airl_reward_alignment")
