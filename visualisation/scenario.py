from visualisation.utils_config import get

def label_scenario(cfg_flat: dict) -> str:
    """Label scenario based on flattened config keys.

    Args:
        cfg_flat: Flattened configuration dictionary

    Returns:
        Scenario label string
    """
    heldout_region = get(cfg_flat, 'eval.heldout_region')
    confounded = get(cfg_flat, 'env.confounded')
    env_name = get(cfg_flat, 'env.name')
    test_z = get(cfg_flat, 'eval.test_z')
    slip_prob = get(cfg_flat, 'env.slip_prob', 0)
    num_trajectories = get(cfg_flat, 'expert.num_trajectories', 1000)
    reward_type = get(cfg_flat, 'env.reward_type', '')

    if heldout_region is not None:
        return 'heldout'
    if confounded or env_name == 'ConfoundedGridWorld':
        if test_z is not None:
            return 'confounded_crossZ'
        return 'confounded'
    if slip_prob and slip_prob > 0:
        return 'noisy'
    if num_trajectories <= 10:
        return 'fewshot'
    if reward_type == 'shaped':
        return 'shaped'
    return 'baseline'
