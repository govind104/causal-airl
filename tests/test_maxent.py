import numpy as np
from envs.gridworld import GridWorld
from irl.maxent_irl import maxent_irl
from envs.utils import build_transition_matrix
from experiments.eval import reward_correlation


def test_maxent_goal_at_corner(seed=0):
    np.random.seed(seed)

    env = GridWorld(grid_size=(5, 5), terminal_states=[(4, 4)], gamma=0.99)
    demos = env.sample_expert_trajectories(n_trajectories=20, optimality="optimal")

    T = build_transition_matrix(grid_size=(5, 5), slip_prob=0.0)
    feature_matrix = env.get_feature_matrix()
    expert_feat_exp = env.get_empirical_feature_expectation(demos)

    n_states = env.n_rows * env.n_cols
    start_dist = np.zeros(n_states)

    # Convert demos into state indices for maxent_irl
    trajectories = []

    for traj in demos:
        traj_indices = []
        s0 = traj[0][0]
        idx = s0[0] * env.n_cols + s0[1]
        start_dist[idx] += 1
        for (s, a, r, s_next) in traj:
            s_idx = s[0] * env.n_cols + s[1]
            traj_indices.append(s_idx)
        trajectories.append(traj_indices)

    start_dist /= len(demos)

    reward, _ = maxent_irl(
        feature_matrix=feature_matrix,
        T=T,
        trajectories=trajectories,
        start_dist=start_dist,
        gamma=0.99,
        learning_rate=0.1,
        n_iters=20,
        verbose=True
    )

    true_reward = env.get_ground_truth_reward()
    corr = reward_correlation(true_reward, reward)
    max_pos = np.unravel_index(np.argmax(reward), reward.shape)

    print(f"[MaxEnt IRL] Reward correlation: {corr:.3f}, max reward at: {max_pos}")

    assert corr > 0.7, "Reward correlation too low"
    assert max_pos == (4, 4), f"Max reward not at goal — got {max_pos}"


if __name__ == "__main__":
    test_maxent_goal_at_corner()
    print("[PASS] test_maxent_goal_at_corner")
