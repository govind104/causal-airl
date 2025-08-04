import argparse
import yaml
import os
import numpy as np
import torch
import random
import gc
import json
import git
from datetime import datetime
from scipy.sparse import issparse

from irl.ng_russell import run_ng_russell, log_memory
from irl.maxent_irl import run_maxent_irl, soft_value_iteration, compute_policy_from_value
from irl.airl import AIRLAgent, create_gridworld_encoder, create_onehot_encoder, create_cartpole_encoder
from irl.causal_airl import CausalAIRLAgent

from envs.environments import BaseEnv, CartPoleWrapper, build_env
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

def save_trajectories(policy, env, n_traj=10, z=None, torch_policy=True, device='cpu', state_encoder=None):
    """Save policy trajectories for visualization and analysis"""
    trajectories = []
    for _ in range(n_traj):
        if z is not None and hasattr(env, 'confounder_values'):
            env.z = z
        s_raw = env.reset()
        s = s_raw[0] if isinstance(s_raw, tuple) else s_raw
        traj = []
        done = False
        while not done and len(traj) < 1000:  # Prevent infinite loops
            if torch_policy:
                if isinstance(s, (int, np.integer)):
                    s_tensor = torch.tensor([[s]], dtype=torch.float32, device=device)
                else:
                    s_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
                s_encoded = state_encoder(s_tensor)
                dist = policy(s_encoded)
                a = dist.probs.argmax(dim=-1).item()
            else:
                a = policy(s)  # For MaxEnt/Ng methods
            step_out = env.step(a)
            if len(step_out) == 5:
                s_next, _, terminated, truncated, _ = step_out
            else:
                s_next, _, terminated, truncated = step_out
            done = terminated or truncated
            traj.append((s, a))
            s = s_next
            if done:
                traj.append((s_next, None))
                break
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
    metrics_to_save = metrics.get_logs() if hasattr(metrics, 'get_logs') else metrics
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
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
        'true_reward': env.get_ground_truth_reward().tolist() if hasattr(env, 'get_ground_truth_reward') else None
    }
    with open(os.path.join(save_dir, 'env_data.json'), 'w') as f:
        json.dump(env_data, f, indent=2)

def run_experiment(cfg):
    """Main experiment runner with unified IRL interface"""
    log_memory("On start")
    set_seed(cfg['train']['seed'])
    
    save_dir = create_run_dir(cfg['eval']['save_dir'], cfg)
    os.makedirs(save_dir, exist_ok=True)
    
    log_memory("Before Environment Building")
    # Build environment and sample demonstrations
    env = build_env(cfg)
    if hasattr(env, "reset_transition_cache"):
        env.reset_transition_cache()
    demos = env.sample_expert_trajectories(
        n_trajectories=cfg['expert']['num_trajectories'],
        optimality=cfg['expert']['optimality'],
        z=cfg['expert'].get('confounder_value', None)
    )
    log_memory("After Environment Building")
    
    # Dispatch to appropriate IRL method
    method = cfg['irl']['method']
    agent = None
    use_sparse = cfg['irl'].get("use_sparse_constraints", False)
    # Safety warning for MaxEnt
    if method == 'maxent' and not use_sparse and env.n_states > 5000:
        print("[Warning] MaxEnt with dense T on large grid may cause memory issues.")

    log_memory("Before Experiments")
    if method == 'ng':
        result = run_ng_russell(cfg, env, demos)
        reward, metrics, additional_data = result
        wrapped_metrics = TrainingLogger()
        wrapped_metrics.log(metrics)
        metrics = wrapped_metrics
        if hasattr(env, 'n_states'): 
            assert reward.shape[0] == env.n_states, f"Reward shape mismatch: {reward.shape[0]} vs {env.n_states}"
    elif method == 'maxent':
        reward, metrics, additional_data = run_maxent_irl(cfg, env, demos)
        wrapped_metrics = TrainingLogger()
        wrapped_metrics.log(metrics)
        metrics = wrapped_metrics
        if hasattr(env, 'n_states'): 
            assert reward.shape[0] == env.n_states, f"Reward shape mismatch: {reward.shape[0]} vs {env.n_states}"
    elif method == 'airl':
        state_encoder = create_cartpole_encoder() if isinstance(env, CartPoleWrapper) else lambda x: x
        action_encoder = create_onehot_encoder(num_classes=env.n_actions)
        if isinstance(env, CartPoleWrapper):
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
        else:
            state_dim = env.n_states
            action_dim = env.n_actions
        agent = AIRLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=cfg['irl']['gamma'],
            lr=cfg['irl']['lr'],
            state_encoder=state_encoder,
            action_encoder=action_encoder
        )
        reward, metrics, agent_data = agent.train(cfg, env, demos)
        print(f"[RUN] {method.upper()} reward shape: {reward.shape}, min={reward.min()}, max={reward.max()}")
        wrapped_metrics = TrainingLogger()
        wrapped_metrics.log(metrics)
        metrics = wrapped_metrics
        if hasattr(env, 'n_states'): 
            assert reward.shape[0] == env.n_states, f"Reward shape mismatch: {reward.shape[0]} vs {env.n_states}"
        additional_data = {
            'reward': reward,
            'policy': agent.policy,
            'discriminator': agent.discriminator,
            **agent_data
        }
    elif method == 'causal_airl':
        state_encoder = create_cartpole_encoder() if isinstance(env, CartPoleWrapper) else create_gridworld_encoder(grid_size=env.grid_size[0])
        action_encoder = create_onehot_encoder(num_classes=env.n_actions)
        if isinstance(env, CartPoleWrapper):
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
        else:
            state_dim = env.n_states
            action_dim = env.n_actions
        agent = CausalAIRLAgent(
            env=env,
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=cfg['irl']['latent_dim'],
            gamma=cfg['irl']['gamma'],
            invariance_penalty=cfg['irl']['invariance_penalty'],
            lr=cfg['irl']['lr'],
            state_encoder=state_encoder,
            action_encoder=action_encoder
        )
        reward, metrics, agent_data = agent.train(cfg, env, demos)
        print(f"[RUN] {method.upper()} reward shape: {reward.shape}, min={reward.min()}, max={reward.max()}")
        wrapped_metrics = TrainingLogger()
        wrapped_metrics.log(metrics)
        metrics = wrapped_metrics
        if hasattr(env, 'n_states'): 
            assert reward.shape[0] == env.n_states, f"Reward shape mismatch: {reward.shape[0]} vs {env.n_states}"
        additional_data = {
            'agent': agent,  # Store agent for later use
            'reward': reward,
            'policy': agent.policy,
            'discriminator': agent.discriminator,
            'encoder': agent.encoder,
            **agent_data
        }
    else:
        raise NotImplementedError(f"Unknown IRL method: {method}")
    
    log_memory("After Experiments")

    # Add environment and final metrics
    additional_data['env'] = env

    log_memory("Before Value Iteration")
    # Skip value iteration if requested
    if not cfg['eval'].get('skip_value_iteration', False):
        # Pass precomputed T if available
        T = additional_data.get('T', None)
        if T is not None:
            print(f"[{method.upper()}] Using {'sparse' if issparse(T) else 'dense'} transition matrix")
        metrics.log(evaluate_irl_result(env, reward, cfg['irl']['gamma'], T=T))
    else:
        metrics.log({
            "reward_correlation": None,
            "value_difference": None,
            "policy_agreement": None,
            "continuous": False})
    log_memory("After Value Iteration")

    # Generate and save policy trajectories
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learned_trajectories = None

    log_memory("Before Trajectory Generation")
    if not cfg['eval'].get('gen_trajectories', True):
        print(f"[Skipping] Trajectory generation for method '{method}' disabled (eval.gen_trajectories={cfg['eval'].get('gen_trajectories')})")
    else:
        if method in ['airl', 'causal_airl']:
            # Generate trajectories using learned policy
            policy = additional_data['policy'].to(device)

            # Determine correct state encoder
            if isinstance(env, CartPoleWrapper):
                state_encoder = create_cartpole_encoder()
            elif hasattr(env, 'grid_size'):
                state_encoder = create_gridworld_encoder(grid_size=env.grid_size[0])
            else:
                raise ValueError("Unknown environment type for state encoder.")

            learned_trajectories = save_trajectories(
                policy, env, n_traj=10, device=device, state_encoder=state_encoder
            )
            np.save(os.path.join(save_dir, 'trajectories.npy'), np.array(learned_trajectories, dtype=object))
            
            # Generate per-Z trajectories for confounded environments
            if hasattr(env, 'confounder_values'):
                for z in env.confounder_values:
                    z_trajs = save_trajectories(
                        policy, env, n_traj=10, z=z, device=device, state_encoder=state_encoder
                    )
                    np.save(os.path.join(save_dir, f'trajectories_z{z}.npy'), np.array(z_trajs, dtype=object))
        
        # For maxent and ng methods (only in GridWorld)
        elif method in ['maxent', 'ng'] and hasattr(env, 'grid_size'):
            # Reuse precomputed T if available
            T = additional_data.get('T', env.build_transition_matrix(sparse=use_sparse))
            
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
            np.save(os.path.join(save_dir, 'trajectories.npy'), np.array(learned_trajectories, dtype=object))
            
            # Generate per-Z trajectories for confounded environments
            if hasattr(env, 'confounder_values'):
                for z in env.confounder_values:
                    z_trajs = save_trajectories(
                        policy_fn, env, n_traj=10, z=z, torch_policy=False
                    )
                    np.save(os.path.join(save_dir, f'trajectories_z{z}.npy'), np.array(z_trajs, dtype=object))
    log_memory("After Trajectory Generation")

    # Compute and log additional metrics
    new_metrics = {}
    
    log_memory("Before Metric Calculation and Saving")
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
    
    metrics.log(new_metrics)
    
    # Save results
    save_experiment_results(save_dir, cfg, metrics, additional_data)
    log_memory("After Metric Calculation and Saving")

    log_memory("Before Memory Cleanup")
    # Clean up
    del env, demos, reward, metrics
    if 'agent' in additional_data:
        del additional_data['agent']
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_memory("After Memory Cleanup")

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
