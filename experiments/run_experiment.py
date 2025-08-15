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
from experiments.eval import evaluate_irl_result, compute_trajectory_overlap

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

def _merge_logs_into_logger(logger: TrainingLogger, logs_dict: dict):
    """
    Safely merge a dict-of-lists (from agent.get_logs()) into a TrainingLogger,
    preserving all per-iteration values.
    """
    for k, v in (logs_dict or {}).items():
        if isinstance(v, list):
            for val in v:
                logger.log(k, val)

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
            s_next, _, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            traj.append((s, a))
            s = s_next
            if done:
                traj.append((s_next, None))
                break
        trajectories.append(traj)
    return trajectories

def _jsonify(obj):
    """Convert non-JSON-serializable objects to JSON-serializable equivalents."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    elif isinstance(obj, set):
        return [_jsonify(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item') and callable(obj.item):  # numpy scalar
        return obj.item()
    else:
        return obj

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
    if 'action_dim' in additional_data:
        cfg['irl']['action_dim'] = additional_data['action_dim']

    def sanitize_json(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: sanitize_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_json(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray) and obj.size == 1:
            return float(obj.item())
        else:
            return obj

    # Sanitize and save config
    sanitized_cfg = sanitize_json(cfg)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(_jsonify(sanitized_cfg), f, indent=2)

    # Write flattened mirror for visualisation (dot-keys)
    def _flatten(nested, sep=".", prefix=""):
        flat = {}
        for k, v in nested.items():
            key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(_flatten(v, sep, key))
            else:
                flat[key] = v
        return flat
    with open(os.path.join(save_dir, 'config_flat.json'), 'w') as f:
        json.dump(_flatten(sanitized_cfg), f, indent=2)

    # 2. Save metrics
    metrics_to_save = metrics.get_logs() if hasattr(metrics, 'get_logs') else metrics
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_to_save, f, indent=2)

    # Ensure training_logs.json exists for CartPole compatibility
    method = cfg['irl']['method']
    env_name = cfg['env']['name']
    if env_name == "CartPole" and method in ['airl', 'causal_airl']:
        training_logs_path = os.path.join(save_dir, 'training_logs.json')
        if not os.path.exists(training_logs_path):
            # Save metrics as training_logs.json for downstream compatibility
            with open(training_logs_path, 'w') as f:
                json.dump(metrics_to_save, f, indent=2)
            print(f"[INFO] Created training_logs.json for CartPole compatibility")

    # 3. Save reward data (if applicable)
    if 'reward' in additional_data and additional_data['reward'] is not None:
        reward = additional_data['reward']
        np.save(os.path.join(save_dir, 'learned_reward.npy'), reward)

        # Also save 2D map for GridWorld visualization (save-time only)
        env_obj = additional_data.get('env', None)
        if env_obj is not None and hasattr(env_obj, 'grid_size'):
            H, W = env_obj.grid_size
            if reward.size == H * W:
                reward_map = reward.reshape(H, W)
                np.save(os.path.join(save_dir, 'learned_reward_map.npy'), reward_map)

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
                inv_reward = additional_data['invariant_reward']
                np.save(os.path.join(save_dir, 'invariant_reward.npy'), inv_reward)

                # Save 2D map for GridWorld visualization
                env_obj = additional_data.get('env', None)
                if env_obj is not None and hasattr(env_obj, 'grid_size'):
                    H, W = env_obj.grid_size
                    if inv_reward.size == H * W:
                        np.save(os.path.join(save_dir, 'invariant_reward_map.npy'), inv_reward.reshape(H, W))

            if 'causal_reward' in additional_data:
                causal_reward = additional_data['causal_reward']
                np.save(os.path.join(save_dir, 'causal_reward.npy'), causal_reward)

                # Save 2D map for GridWorld visualization
                env_obj = additional_data.get('env', None)
                if env_obj is not None and hasattr(env_obj, 'grid_size'):
                    H, W = env_obj.grid_size
                    if causal_reward.size == H * W:
                        np.save(os.path.join(save_dir, 'causal_reward_map.npy'), causal_reward.reshape(H, W))
            
            # Save per-Z reward maps for confounded environments
        save_per_z = cfg.get('eval', {}).get('save_per_z', False) or cfg.get('irl', {}).get('num_z_samples', 0) > 1
        if save_per_z and hasattr(additional_data['env'], 'confounder_values'):
                if 'invariant_reward' not in additional_data or 'causal_reward' not in additional_data:
                    print(f"Warning: Skipping per-Z reward saving - causal/invariant reward missing")
                else:
                    # Create per_z subdirectory
                    per_z_dir = os.path.join(save_dir, 'per_z')
                    os.makedirs(per_z_dir, exist_ok=True)

                    for i, z_value in enumerate(additional_data['env'].confounder_values):
                        if 'agent' in additional_data:
                            try:
                                reward_z, _, _ = additional_data['agent'].extract_reward_components_for_z(
                                    additional_data['env'], z_value
                                )
                            except AttributeError:
                                print(f"Warning: Agent type {type(additional_data['agent']).__name__} does not support per-z reward extraction")
                                print("Skipping per-z reward saving for this agent type")
                                break

                            # Save 1D reward vector (preserving contract)
                            reward_z_flat = np.asarray(reward_z).flatten()
                            np.save(os.path.join(per_z_dir, f'reward_z{i:03d}.npy'), reward_z_flat)

                            # Save 2D reward map for visualization (reshape at save-time only)
                            env_obj = additional_data.get('env', None)
                            if env_obj is not None and hasattr(env_obj, 'grid_size'):
                                H, W = env_obj.grid_size
                                if reward_z_flat.size == H * W:
                                    reward_map_2d = reward_z_flat.reshape(H, W)
                                    np.save(os.path.join(per_z_dir, f'reward_map_z{i:03d}.npy'), reward_map_2d)
                            print(f"[INFO] Saved per-z rewards for z={z_value} as z{i:03d}")
                        else:
                            print("Warning: extract_reward_components_for_z not available - skipping per-Z reward saving")
        
        if method in ['airl', 'causal_airl']:
            # Save training logs
            if 'training_logs' in additional_data and hasattr(additional_data['training_logs'], "save"):
                additional_data['training_logs'].save(os.path.join(save_dir, 'training_logs.json'))

                # Optional alias for older scripts
                try:
                    src = os.path.join(save_dir, 'training_logs.json')
                    dst = os.path.join(save_dir, 'training_log.json')
                    with open(src, 'r') as _s, open(dst, 'w') as _d:
                        _d.write(_s.read())
                except Exception:
                    pass

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
    # Ensure terminal_states is serialized as list of lists
    env_data = {
        'name': env_name,
        'grid_size': env.grid_size if hasattr(env, 'grid_size') else None,
        'terminal_states': list(env.terminal_states) if hasattr(env, 'terminal_states') else None,
        'true_reward': env.get_ground_truth_reward().tolist() if hasattr(env, 'get_ground_truth_reward') else None
    }

    # Add held-out region info if present
    if 'heldout_region' in additional_data:
        env_data['heldout_region'] = additional_data['heldout_region']
    if 'heldout_mask_indices' in additional_data:
        env_data['heldout_mask_indices'] = additional_data['heldout_mask_indices']

    # Add CartPole physics parameters
    if isinstance(env, CartPoleWrapper):
        env_data.update({
            'pole_length': getattr(env.env.env, 'length', 0.5),
            'masscart': getattr(env.env.env, 'masscart', 1.0),
            'masspole': getattr(env.env.env, 'masspole', 0.1),
            'gravity': getattr(env.env.env, 'gravity', 9.8),
            'tau': getattr(env.env.env, 'tau', 0.02)
        })

    with open(os.path.join(save_dir, 'env_data.json'), 'w') as f:
        json.dump(_jsonify(env_data), f, indent=2)

def run_experiment(cfg):
    """Main experiment runner with unified IRL interface"""
    log_memory("On start")
    set_seed(cfg['train']['seed'])

    save_dir = create_run_dir(cfg['eval']['save_dir'], cfg)
    os.makedirs(save_dir, exist_ok=True)

    # Write manifest.json with config, overrides, seed, and git hash
    manifest = {
        'config': cfg.get('config_path', None),
        'seed': cfg.get('train', {}).get('seed', None),
        'start_time': datetime.utcnow().isoformat() + 'Z',
        # Ensure 'overrides' key exists for viz compatibility
        'overrides': cfg.get('overrides', cfg.get('_overrides', {})),
    }
    try:
        repo = git.Repo(search_parent_directories=True)
        manifest['git_hash'] = repo.head.object.hexsha
    except Exception:
        manifest['git_hash'] = None
    with open(os.path.join(save_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    
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
    # Save expert (s, a) pairs for attribution
    expert_states, expert_actions = [], []
    for traj in demos:
        for (s, a, _, _) in traj:
            expert_states.append(s)
            expert_actions.append(a)
    np.save(os.path.join(save_dir, 'states.npy'), np.array(expert_states, dtype=np.float32))
    np.save(os.path.join(save_dir, 'actions.npy'), np.array(expert_actions, dtype=np.int64))
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
        metrics_logger = TrainingLogger()
        _merge_logs_into_logger(metrics_logger, metrics)
        metrics = metrics_logger
        if hasattr(env, 'n_states'): 
            assert reward.shape[0] == env.n_states, f"Reward shape mismatch: {reward.shape[0]} vs {env.n_states}"
    elif method == 'maxent':
        reward, metrics, additional_data = run_maxent_irl(cfg, env, demos)
        metrics_logger = TrainingLogger()
        _merge_logs_into_logger(metrics_logger, metrics)
        metrics = metrics_logger
        if hasattr(env, 'n_states'): 
            assert reward.shape[0] == env.n_states, f"Reward shape mismatch: {reward.shape[0]} vs {env.n_states}"
    elif method == 'airl':
        state_encoder = create_cartpole_encoder() if isinstance(env, CartPoleWrapper) else create_gridworld_encoder(grid_size=env.grid_size[0])
        action_encoder = create_onehot_encoder(num_classes=env.n_actions)
        if isinstance(env, CartPoleWrapper):
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            # Ensure consistency for CartPole
            encoded_dim = state_dim  # Identity encoding for CartPole
        else:
            state_dim = env.n_states
            action_dim = env.n_actions
            # For GridWorld, encoded_dim will be computed via infer_encoded_dim
        agent = AIRLAgent(
            env=env,
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=cfg['irl']['gamma'],
            lr=cfg['irl']['lr'],
            state_encoder=state_encoder,
            action_encoder=action_encoder,
            reward_logit_clip=cfg['irl'].get('reward_logit_clip')
        )
        reward, metrics, agent_data = agent.train(cfg, env, demos)
        print(f"[RUN] {method.upper()} reward shape: {reward.shape}, min={reward.min()}, max={reward.max()}")
        metrics_logger = TrainingLogger()
        _merge_logs_into_logger(metrics_logger, metrics)
        metrics = metrics_logger
        if hasattr(env, 'n_states'): 
            assert reward.shape[0] == env.n_states, f"Reward shape mismatch: {reward.shape[0]} vs {env.n_states}"
        additional_data = {
            'training_logs': metrics,
            'reward': reward,
            'policy': agent.policy,
            'discriminator': agent.discriminator,
            'action_dim': action_dim,
            **agent_data
        }
    elif method == 'causal_airl':
        state_encoder = create_cartpole_encoder() if isinstance(env, CartPoleWrapper) else create_gridworld_encoder(grid_size=env.grid_size[0])
        action_encoder = create_onehot_encoder(num_classes=env.n_actions)
        if isinstance(env, CartPoleWrapper):
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            # Ensure consistency for CartPole
            encoded_dim = state_dim  # Identity encoding for CartPole
        else:
            state_dim = env.n_states
            action_dim = env.n_actions
            # For GridWorld, encoded_dim will be computed via infer_encoded_dim
        agent = CausalAIRLAgent(
            env=env,
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=cfg['irl']['latent_dim'],
            gamma=cfg['irl']['gamma'],
            invariance_penalty=cfg['irl']['invariance_penalty'],
            lr=cfg['irl']['lr'],
            state_encoder=state_encoder,
            action_encoder=action_encoder,
            reward_logit_clip=cfg['irl'].get('reward_logit_clip')
        )
        reward, metrics, agent_data = agent.train(cfg, env, demos)
        print(f"[RUN] {method.upper()} reward shape: {reward.shape}, min={reward.min()}, max={reward.max()}")
        metrics_logger = TrainingLogger()
        _merge_logs_into_logger(metrics_logger, metrics)
        metrics = metrics_logger
        if hasattr(env, 'n_states'): 
            assert reward.shape[0] == env.n_states, f"Reward shape mismatch: {reward.shape[0]} vs {env.n_states}"
        additional_data = {
            'training_logs': metrics,
            'agent': agent,  # Store agent for later use
            'reward': reward,
            'policy': agent.policy,
            'discriminator': agent.discriminator,
            'encoder': agent.encoder,
            'action_dim': action_dim,
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
        # Simply aggregate final summary statistics from per-iteration logs for CartPole
        env_name = cfg['env']['name']
        if env_name == 'CartPole':
            logs = metrics.get_logs() if hasattr(metrics, 'get_logs') else metrics
            # Collect per-iteration arrays
            avg_returns = logs.get('avg_episode_return', [])
            avg_lens = logs.get('avg_episode_length', [])
            ret_stds = logs.get('episode_return_std', [])
            # Compute averaged final summary metrics
            if len(avg_returns) > 0:
                aggregate_metrics = {
                    'avg_return': float(np.mean(avg_returns)),
                    'avg_length': float(np.mean(avg_lens)) if len(avg_lens) > 0 else None,
                    'return_std': float(np.mean(ret_stds)) if len(ret_stds) > 0 else None,
                    'continuous': True
                }
                # Log the aggregate summary metrics
                metrics.log(aggregate_metrics)
        else:
            # Pass precomputed T if available
            learned_state_reward = np.asarray(reward, dtype=float).reshape(-1)
            print(f"[DEBUG] Learned reward vector shape: {learned_state_reward.shape}")
            T = additional_data.get('T', None)
            if T is not None:
                print(f"[{method.upper()}] Using {'sparse' if issparse(T) else 'dense'} transition matrix")

            # Optional held-out state masking for generalisation tests
            heldout_mask = None
            region = None
            heldout_region_config = cfg.get('eval', {}).get('heldout_region')
            if heldout_region_config:
                region = heldout_region_config
                if hasattr(env, 'grid_size') and hasattr(env, 'state_to_index'):
                    H, W = env.grid_size
                    heldout_mask = np.zeros(env.n_states, dtype=bool)
                    def in_region(i,j):
                        if region == 'top_left': return i < H//2 and j < W//2
                        if region == 'top_right': return i < H//2 and j >= W//2
                        if region == 'bottom_left': return i >= H//2 and j < W//2
                        if region == 'bottom_right': return i >= H//2 and j >= W//2
                        return False
                    for i_ in range(H):
                        for j_ in range(W):
                            idx = env.state_to_index((i_, j_), n_cols=W)
                            if in_region(i_, j_):
                                heldout_mask[idx] = True

                    # Store held-out region info for later saving
                    additional_data['heldout_region'] = region
                    additional_data['heldout_mask_indices'] = heldout_mask.nonzero()[0].tolist()

                else:
                    # If no held-out region is configured, region stays None and we don't need any warning
                    print(f"[Warning] Held-out region '{region}' specified but environment doesn't support it")

            eval_results = evaluate_irl_result(env, cfg, save_dir, learned_state_reward, cfg['irl']['gamma'], T=T, heldout_mask=heldout_mask)

            # Ensure all environments log evaluation results consistently
            if eval_results is not None:
                metrics.log(eval_results)

        # Save value function for CartPole if requested
        if isinstance(env, CartPoleWrapper) and cfg.get('eval', {}).get('save_value_function', False):
            try:
                # For CartPole, estimate value function via policy rollouts
                policy = additional_data.get('policy')
                if policy is not None:
                    # Approximate V as average episode returns over state space
                    V_approx = np.array([eval_results.get('avg_episode_return', 0.0)])
                    np.save(os.path.join(save_dir, 'V.npy'), V_approx)
                    print(f"[INFO] Saved CartPole value function approximation")
            except Exception as e:
                print(f"[Warning] Failed to save CartPole value function: {e}")

    else:
        metrics.log({
            "reward_correlation": None,
            "value_difference": None,
            "policy_agreement": None,
            "continuous": False})
    log_memory("After Value Iteration")

    # Save per-Z reward maps for confounded environments BEFORE final save
    if method == 'causal_airl' and hasattr(env, 'confounder_values') and env.confounder_values:
        per_z_dir = os.path.join(save_dir, 'per_z')
        os.makedirs(per_z_dir, exist_ok=True)
        agent = additional_data.get('agent', None)
        if agent and hasattr(agent, 'extract_reward_components_for_z'):
            for i, z_value in enumerate(env.confounder_values):
                try:
                    reward_z, _, _ = agent.extract_reward_components_for_z(env, z_value)
                    reward_z_flat = np.asarray(reward_z).flatten()
                    np.save(os.path.join(per_z_dir, f'reward_z{i:03d}.npy'), reward_z_flat)
                    # Save 2D map if GridWorld
                    if hasattr(env, 'grid_size'):
                        H, W = env.grid_size
                        if reward_z_flat.size == H * W:
                            np.save(os.path.join(per_z_dir, f'reward_map_z{i:03d}.npy'), reward_z_flat.reshape(H, W))
                    print(f"[INFO] Saved per-z rewards for z={z_value} as z{i:03d}")
                except Exception as e:
                    print(f"[Warning] Failed to save per-z reward for z={z_value}: {e}")

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
        new_metrics["reward_variance"] = additional_data.get("reward_var_z", None)
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
