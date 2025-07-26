import numpy as np
import torch

from envs.gridworld import GridWorld
from envs.utils import build_transition_matrix, trajectory_to_tensor_batch, greedy_policy_from_map
from models.policy import PolicyNet
from irl.airl import AIRLDiscriminator
from experiments.eval import reward_correlation, policy_agreement
from experiments.run_experiment import collect_agent_rollouts


def test_airl_reward_recovery(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = GridWorld(grid_size=(5, 5), terminal_states=[(4, 4)], gamma=0.99)
    demos = env.sample_expert_trajectories(n_trajectories=5, optimality="optimal")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNet(state_dim=25, action_dim=4).to(device)
    disc = AIRLDiscriminator(state_dim=25, action_dim=4).to(device)
    optimizer_d = torch.optim.Adam(disc.parameters(), lr=1e-3)
    optimizer_pi = torch.optim.Adam(policy.parameters(), lr=1e-2)

    for it in range(5):
        # Sample expert batch
        s_e, a_e, s_pe = trajectory_to_tensor_batch(demos, device)

        # Rollout agent
        agent_data = collect_agent_rollouts(policy, env, 4, device, episodes=2)
        s_pi, a_pi, s_ppi, logp = map(torch.stack, zip(*agent_data))

        # Update discriminator
        optimizer_d.zero_grad()
        expert_preds = disc(s_e, a_e, s_pe)
        agent_preds = disc(s_pi, a_pi, s_ppi)
        labels = torch.cat([torch.ones_like(expert_preds), torch.zeros_like(agent_preds)])
        preds = torch.cat([expert_preds, agent_preds])
        loss_d = torch.nn.functional.binary_cross_entropy(torch.sigmoid(preds), labels)
        loss_d.backward()
        optimizer_d.step()

        # Update policy with AIRL reward
        with torch.no_grad():
            rewards = disc.reward(s_pi, a_pi, s_ppi)

        returns = rewards  # no baseline
        loss_pi = -(logp * returns.squeeze()).mean()

        optimizer_pi.zero_grad()
        loss_pi.backward()
        optimizer_pi.step()

    # Evaluate learned reward
    reward_map = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            s = torch.zeros(1, 25).to(device)
            s[0, i * 5 + j] = 1.0
            a = torch.tensor([1]).to(device)  # arbitrary
            a_onehot = torch.nn.functional.one_hot(a, num_classes=4).float()
            s_prime = s  # dummy next state
            r = disc.reward(s, a_onehot, s_prime).item()
            reward_map[i, j] = r

    true_reward = env.get_ground_truth_reward()
    corr = reward_correlation(true_reward, reward_map)
    pi_learned = greedy_policy_from_map(reward_map, env)
    pi_expert = env.get_optimal_policy().reshape((env.n_rows, env.n_cols))
    agree = policy_agreement(pi_expert, pi_learned)

    print(f"[AIRL] Reward corr: {corr:.3f}, Policy agreement: {agree:.3f}")
    assert corr > 0.5, f"AIRL reward does not align well with ground truth: corr = {corr:.3f}"
    assert agree > 0.7, f"AIRL learned policy does not match expert: agreement = {agree:.3f}"

if __name__ == "__main__":
    test_airl_reward_recovery()
    print("[PASS] test_airl_reward_recovery")
