import argparse
import yaml
import os
import numpy as np
import torch
import random
import gc
import json
from datetime import datetime
import git

from irl.ng_russell import run_ng_russell
from irl.maxent_irl import run_maxent_irl, soft_value_iteration, compute_policy_from_value
from irl.airl import AIRLAgent
from irl.causal_airl import CausalAIRLAgent

from envs.environments import BaseEnv, build_env, build_transition_matrix
from experiments.logger import TrainingLogger
from experiments.eval import evaluate_irl_result, compute_trajectory_overlap, compute_reward_variance

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def create_run_dir(base_path, cfg):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method = cfg['irl']['method']
    env_name = cfg['env']['name']
    return f"{base_path}/{method}_{env_name}_{timestamp}"

def save_trajectories(policy, env, n_traj=10, z=None, torch_policy=True, device='cpu'):
    """Save policy trajectories for visualization and analysis"""
    trajectories = []
    for _ in range(n_traj):
        if z is not None and hasattr(env, 'confounder_values'):
            env.z = z
        s = env.reset()
        traj = []
        done = False
        while not done:
            if torch_policy:
                s_tensor = torch.FloatTensor(s).to(device)
                a = policy(s_tensor).argmax().item()
            else:
                a = policy(s)  # For maxent/ng methods
            s_next, _, done, _ = env.step(a)
            traj.append((s, a))
            s = s_next
        trajectories.append(traj)
    return trajectories

def save_experiment_results(save_dir, cfg, metrics, additional_data):
    """Save all experiment results in structured format"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Add git commit hash for reproducibility
    try:
        repo = git.Repo(search_parent_directories=True)
        cfg['git_commit'] = repo.head.object.hexsha
    except:
        cfg['git_commit'] = "unknown"
    
    # 1. Save configuration
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    
    # 2. Save metrics
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 3. Save reward data (if applicable)
    if 'reward' in additional_data and additional_data['reward'] is not None:
        np.save(os.path.join(save_dir, 'learned_reward.npy'), additional_data['reward'])
    
    # 4. Method-specific saves
    method = cfg['irl']['method']
    env_name = cfg['env']['name']
    
    # Handle CartPole specially - save model weights
    if env_name == "CartPole" and method in ['airl', 'causal_airl']:
        model_data = {
            'discriminator': additional_data['discriminator'].state_dict(),
            'policy': additional_data['policy'].state_dict(),
            'config': cfg
        }
        if method == 'causal_airl' and 'encoder' in additional_data:
            model_data['encoder'] = additional_data['encoder'].state_dict()
        torch.save(model_data, os.path.join(save_dir, 'model_weights.pt'))
    
    # For non-CartPole environments
    else:
        if method == 'causal_airl':
            # Save invariant and causal components
            if 'invariant_reward' in additional_data:
                np.save(os.path.join(save_dir, 'invariant_reward.npy'), additional_data['invariant_reward'])
            if 'causal_reward' in additional_data:
                np.save(os.path.join(save_dir, 'causal_reward.npy'), additional_data['causal_reward'])
            
            # Save per-Z reward maps for confounded environments
            if hasattr(additional_data['env'], 'confounder_values'):
                if 'invariant_reward' not in additional_data or 'causal_reward' not in additional_data:
                    print(f"Warning: Skipping per-Z reward saving - causal/invariant reward missing")
                else:
                    for z_value in additional_data['env'].confounder_values:
                        if 'agent' in additional_data:
                            reward_z, _, _ = additional_data['agent'].extract_reward_components_for_z(
                                additional_data['env'], z_value
                            )
                            np.save(os.path.join(save_dir, f'reward_map_z{z_value}.npy'), reward_z)
                        else:
                            print("Warning: extract_reward_components_for_z not available - skipping per-Z reward saving")
        
        if method in ['airl', 'causal_airl']:
            # Save training logs
            if 'training_logs' in additional_data:
                with open(os.path.join(save_dir, 'training_logs.json'), 'w') as f:
                    json.dump(additional_data['training_logs'], f, indent=2)
            
            # Save models
            if 'policy' in additional_data:
                torch.save(additional_data['policy'].state_dict(), 
                           os.path.join(save_dir, 'policy.pt'))
            if 'discriminator' in additional_data:
                torch.save(additional_data['discriminator'].state_dict(), 
                           os.path.join(save_dir, 'discriminator.pt'))
            if method == 'causal_airl' and 'encoder' in additional_data:
                torch.save(additional_data['encoder'].state_dict(), 
                           os.path.join(save_dir, 'encoder.pt'))
    
    # 5. Environment data for visualization
    env = additional_data['env']
    env_data = {
        'name': env_name,
        'grid_size': env.grid_size if hasattr(env, 'grid_size') else None,
        'terminal_states': env.terminal_states if hasattr(env, 'terminal_states') else None,
        'true_reward': env.get_ground_truth_reward() if hasattr(env, 'get_ground_truth_reward') else None
    }
    with open(os.path.join(save_dir, 'env_data.json'), 'w') as f:
        json.dump(env_data, f, indent=2)

def run_experiment(config_path):
    """Main experiment runner with unified IRL interface"""
    cfg = load_config(config_path)
    set_seed(cfg['train']['seed'])
    
    save_dir = create_run_dir(cfg['eval']['save_dir'], cfg)
    os.makedirs(save_dir, exist_ok=True)
    
    # Build environment and sample demonstrations
    env = build_env(cfg)
    demos = env.sample_expert_trajectories(
        n_trajectories=cfg['expert']['num_trajectories'],
        optimality=cfg['expert']['optimality'],
        z=cfg['expert'].get('confounder_value', None)
    )
    
    # Dispatch to appropriate IRL method
    method = cfg['irl']['method']
    agent = None
    if method == 'ng':
        reward, metrics, additional_data = run_ng_russell(cfg, env, demos)
    elif method == 'maxent':
        reward, metrics, additional_data = run_maxent_irl(cfg, env, demos)
    elif method == 'airl':
        agent = AIRLAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            gamma=cfg['irl']['gamma'],
            lr=cfg['irl']['lr']
        )
        reward, metrics, agent_data = agent.train(cfg, env, demos)
        additional_data = {
            'policy': agent.policy,
            'discriminator': agent.discriminator,
            **agent_data
        }
    elif method == 'causal_airl':
        agent = CausalAIRLAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            latent_dim=cfg['irl']['latent_dim'],
            gamma=cfg['irl']['gamma'],
            invariance_penalty=cfg['irl']['invariance_penalty'],
            lr=cfg['irl']['lr']
        )
        reward, metrics, agent_data = agent.train(cfg, env, demos)
        additional_data = {
            'agent': agent,  # Store agent for later use
            'policy': agent.policy,
            'discriminator': agent.discriminator,
            'encoder': agent.encoder,
            **agent_data
        }
    else:
        raise NotImplementedError(f"Unknown IRL method: {method}")
    
    # Add environment and final metrics
    additional_data['env'] = env
    metrics.update(evaluate_irl_result(
        env, reward, env.get_optimal_policy(), cfg['irl']['gamma']
    ))
    
    # Generate and save policy trajectories
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learned_trajectories = None
    
    if method in ['airl', 'causal_airl']:
        # Generate trajectories using learned policy
        policy = additional_data['policy'].to(device)
        learned_trajectories = save_trajectories(
            policy, env, n_traj=10, device=device
        )
        np.save(os.path.join(save_dir, 'trajectories.npy'), learned_trajectories)
        
        # Generate per-Z trajectories for confounded environments
        if hasattr(env, 'confounder_values'):
            for z in env.confounder_values:
                z_trajs = save_trajectories(
                    policy, env, n_traj=10, z=z, device=device
                )
                np.save(os.path.join(save_dir, f'trajectories_z{z}.npy'), z_trajs)
    
    # For maxent and ng methods (only in GridWorld)
    elif method in ['maxent', 'ng'] and hasattr(env, 'grid_size'):
        # Build transition matrix
        T = build_transition_matrix(env.grid_size, env.slip_prob)
        
        # Compute value function and policy
        V = soft_value_iteration(T, reward, cfg['irl']['gamma'])
        pi = compute_policy_from_value(T, reward, V, cfg['irl']['gamma'])
        
        # Create policy function
        def policy_fn(s):
            if isinstance(s, tuple):  # Handle (row, col) states
                idx = env.state_to_index(s, env.n_cols)
            else:  # Handle index states
                idx = s
            return np.argmax(pi[idx])
        
        # Generate trajectories
        learned_trajectories = save_trajectories(
            policy_fn, env, n_traj=10, torch_policy=False
        )
        np.save(os.path.join(save_dir, 'trajectories.npy'), learned_trajectories)
        
        # Generate per-Z trajectories for confounded environments
        if hasattr(env, 'confounder_values'):
            for z in env.confounder_values:
                z_trajs = save_trajectories(
                    policy_fn, env, n_traj=10, z=z, torch_policy=False
                )
                np.save(os.path.join(save_dir, f'trajectories_z{z}.npy'), z_trajs)
    
    # Compute and log additional metrics
    new_metrics = {}
    
    # 1. Reward MSE (only for environments with ground truth reward)
    if hasattr(env, 'get_ground_truth_reward') and reward is not None:
        true_reward = env.get_ground_truth_reward().flatten()
        if reward.shape != true_reward.shape:
            reward = reward.reshape(true_reward.shape)
        new_metrics["reward_mse"] = np.mean((true_reward - reward) ** 2)
    else:
        new_metrics["reward_mse"] = None
    
    # 2. Trajectory overlap (if trajectories were generated)
    if learned_trajectories is not None:
        new_metrics["trajectory_overlap"] = compute_trajectory_overlap(demos, learned_trajectories)
    else:
        new_metrics["trajectory_overlap"] = None
    
    # 3. Reward variance (only for causal_airl with confounder)
    if method == 'causal_airl' and 'causal_reward' in additional_data:
        new_metrics["reward_variance"] = compute_reward_variance(additional_data)
    else:
        new_metrics["reward_variance"] = None
    
    metrics.update(new_metrics)
    
    # Save results
    save_experiment_results(save_dir, cfg, metrics, additional_data)
    
    # Clean up
    del env, demos, reward, metrics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def update_config_with_overrides(cfg, overrides):
    for override in overrides:
        path, value = override.split('=', 1)
        keys = path.split('.')
        current = cfg
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = yaml.safe_load(value)
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", action='append', default=[],
                        help="Override config values (e.g., 'irl.gamma=0.95')")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    cfg = update_config_with_overrides(cfg, args.override)
    run_experiment(cfg)