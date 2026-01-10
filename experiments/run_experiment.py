import argparse
import yaml
import os
import numpy as np
import torch
import random
import gc
import json
import glob

from datetime import datetime
from scipy.sparse import issparse, save_npz

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

def _compute_state_reward_from_airl(discriminator, state_encoder, env):
    """
    Derive a *state-only* reward vector from AIRL by averaging r(s,a) over actions.
    This strips the potential term h, making it suitable for correlation
    with ground-truth *state* rewards. Returns np.ndarray of shape (n_states,).
    """
    discriminator.eval()
    # Enumerate all states as flat indices → one-hot via provided encoder
    idxs = np.arange(env.n_states, dtype=np.int64)
    states_idx = torch.tensor(idxs.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        s_enc = state_encoder(states_idx)                              # [S, dimS]
        actions = torch.arange(env.n_actions, dtype=torch.long)
        a_onehot = torch.nn.functional.one_hot(actions, num_classes=env.n_actions).float()  # [A, dimA]
        s_rep = s_enc.unsqueeze(1).repeat(1, env.n_actions, 1)         # [S, A, dimS]
        a_rep = a_onehot.unsqueeze(0).repeat(env.n_states, 1, 1)       # [S, A, dimA]
        sa = torch.cat([s_rep, a_rep], dim=-1)                         # [S, A, dimS+dimA]
        r_sa = discriminator.r(sa).squeeze(-1)                         # [S, A]
        r_state = r_sa.mean(dim=1)                                     # [S]
    return r_state.cpu().numpy()

def _infer_eval_max_steps(env, cfg):
    """Infer a sensible eval horizon.
    Priority:
      1) cfg['eval']['max_steps'] if provided
      2) GridWorld-style heuristic (slip-aware, grid-size dependent)
      3) Gym env cap if available (env._max_episode_steps or spec.max_episode_steps)
      4) Fallback 1000
    """
    # 1) Explicit override
    try:
        ms = int(cfg.get('eval', {}).get('max_steps', 0))
        if ms > 0:
            return ms
    except Exception:
        pass
    # 2) GridWorld heuristic (mirrors environments.py)
    try:
        if hasattr(env, 'grid_size') and hasattr(env, 'slip_prob'):
            H, W = env.grid_size
            min_path = (W - 1) + (H - 1)
            slowdown = 1.0 / max(1.0 - float(getattr(env, 'slip_prob', 0.0)), 1e-6)
            H_cap = max(100, int(W) * int(H))
            return int(min(slowdown * (min_path + (W + H) + 10), H_cap))
    except Exception:
        pass
    # 3) Gym caps
    try:
        if hasattr(env, '_max_episode_steps') and env._max_episode_steps:
            return int(env._max_episode_steps)
    except Exception:
        pass
    try:
        if hasattr(env, 'spec') and env.spec and getattr(env.spec, 'max_episode_steps', None):
            return int(env.spec.max_episode_steps)
    except Exception:
        pass
    # 4) Fallback
    return 1000

def save_trajectories(policy, env, n_traj=10, z=None, torch_policy=True, device='cpu', state_encoder=None, max_steps: int = 1000):
    """Save policy trajectories for visualisation and analysis.
    Args:
        policy: torch policy (when torch_policy=True) or callable mapping s->a
        env: environment
        n_traj: number of episodes to generate
        z: optional confounder value to set on the env
        torch_policy: whether 'policy' is a torch module returning a dist
        device: torch device
        state_encoder: encoder used before passing state to policy (torch path)
        max_steps: hard cap on per-episode steps (was 1000; now configurable)
    """
    trajectories = []
    for _ in range(n_traj):
        s_raw = env.reset()
        if z is not None:
            if hasattr(env, 'set_confounder'):
                env.set_confounder(z)           # CartPole
            elif hasattr(env, 'z'):
                env.z = z                       # ConfoundedGridWorld

        s = s_raw[0] if isinstance(s_raw, tuple) else s_raw
        traj = []
        done = False
        steps = 0
        while not done and steps < max_steps:
            if torch_policy:
                if isinstance(s, (int, np.integer)):
                    s_tensor = torch.tensor([[s]], dtype=torch.float32, device=device)
                else:
                    s_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
                s_encoded = state_encoder(s_tensor)
                # inference path: avoid tracking grads during evaluation/trajectory saving
                with torch.no_grad():
                    dist = policy(s_encoded)
                a = dist.probs.argmax(dim=-1).item()
            else:
                a = policy(s)  # For MaxEnt/Ng methods
            step_out = env.step(a)
            # Gymnasium-style (5-tuple): (obs, reward, terminated, truncated, info)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                s_next, _, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                # Tabular GridWorld (4-tuple): (state, reward, done, info)
                s_next, _, done, _ = step_out
            traj.append((s, a))
            s = s_next
            steps += 1
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
    
    # Ensure we always have a logger-like object
    if metrics is None:
        metrics = TrainingLogger()

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

        # Also save 2D map for GridWorld visualisation (save-time only)
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

                # Save 2D map for GridWorld visualisation
                env_obj = additional_data.get('env', None)
                if env_obj is not None and hasattr(env_obj, 'grid_size'):
                    H, W = env_obj.grid_size
                    if inv_reward.size == H * W:
                        np.save(os.path.join(save_dir, 'invariant_reward_map.npy'), inv_reward.reshape(H, W))

            if 'causal_reward' in additional_data:
                causal_reward = additional_data['causal_reward']
                np.save(os.path.join(save_dir, 'causal_reward.npy'), causal_reward)

                # Save 2D map for GridWorld visualisation
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

                            # Save 2D reward map for visualisation (reshape at save-time only)
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
    
    # 5. Environment data for visualisation
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
    
    log_memory("Before Environment Building")

    # Build environment and sample demonstrations
    env = build_env(cfg)

    # Create logger up front
    metrics = TrainingLogger()

    # Apply held-out region filtering to training data
    heldout_region = cfg.get('eval', {}).get('heldout_region')
    training_heldout_mask = None
    if heldout_region and hasattr(env, 'grid_size') and hasattr(env, 'state_to_index'):
        H, W = env.grid_size
        training_heldout_mask = np.zeros(env.n_states, dtype=bool)

        def in_region(i,j):
            if heldout_region == 'top_left': return i < H//2 and j < W//2
            if heldout_region == 'top_right': return i < H//2 and j >= W//2
            if heldout_region == 'bottom_left': return i >= H//2 and j < W//2
            if heldout_region == 'bottom_right': return i >= H//2 and j >= W//2
            return False

        for i_ in range(H):
            for j_ in range(W):
                idx = env.state_to_index((i_, j_), n_cols=W)
                if in_region(i_, j_):
                    training_heldout_mask[idx] = True

    if hasattr(env, "reset_transition_cache"):
        env.reset_transition_cache()

    demos_raw = env.sample_expert_trajectories(
        n_trajectories=cfg['expert']['num_trajectories'],
        optimality=cfg['expert']['optimality'],
        z=cfg['expert'].get('confounder_value', None)
    )

    # Filter training demos to exclude held-out region
    if training_heldout_mask is not None:
        demos = []
        dropped_samples = 0
        for traj in demos_raw:
            filtered_traj = []
            for (s, a, r, s_p) in traj:
                idx = env.state_to_index(s, n_cols=W)
                if not training_heldout_mask[idx]:
                    filtered_traj.append((s, a, r, s_p))
                else:
                    dropped_samples += 1
            if filtered_traj:
                demos.append(filtered_traj)
        print(f"[Info] Filtered {dropped_samples} samples from held-out region '{heldout_region}'")
    else:
        demos = demos_raw

    # = Sample-efficiency bookkeeping (log planned subset sizes) =
    try:
        n = len(demos)
        demo_counts = sorted(set([max(1, n//4), max(1, n//2), max(1, 3*n//4), n]))
        metrics.log({f"demo_count_{k}": k for k in demo_counts})
    except Exception as _e:
        print(f"[WARN] demo-count logging failed: {_e}")

    # Handle test_z for cross-confounder evaluation
    test_z = cfg['eval'].get('test_z', None)
    if test_z is not None and hasattr(env, 'confounder_values'):
        print(f"[Info] Using test_z={test_z} for evaluation (trained on different confounder)")

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

    # Baseline floor metrics (namespaced)
    baselines = {}
    if hasattr(env, 'n_actions'):
        baselines['baselines/random_policy_agreement'] = 1.0/float(env.n_actions)
    if hasattr(env, 'get_ground_truth_reward') and hasattr(env, 'n_states'):
        gt = env.get_ground_truth_reward().ravel()
        rnd = np.random.randn(env.n_states)
        baselines['baselines/random_reward_corr'] = float(np.corrcoef(gt, rnd)[0,1])
    metrics.log(baselines)

    if method == 'ng':
        result = run_ng_russell(cfg, env, demos)
        reward, new_metrics, additional_data = result
        _merge_logs_into_logger(metrics, new_metrics)
        if hasattr(env, 'n_states'): 
            assert reward.shape[0] == env.n_states, f"Reward shape mismatch: {reward.shape[0]} vs {env.n_states}"
    elif method == 'maxent':
        reward, new_metrics, additional_data = run_maxent_irl(cfg, env, demos)
        _merge_logs_into_logger(metrics, new_metrics)
        if hasattr(env, 'n_states'): 
            assert reward.shape[0] == env.n_states, f"Reward shape mismatch: {reward.shape[0]} vs {env.n_states}"
    elif method == 'airl':
        state_encoder = create_cartpole_encoder() if isinstance(env, CartPoleWrapper) else create_gridworld_encoder(grid_size=env.grid_size[1])
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
        reward, new_metrics, agent_data = agent.train(cfg, env, demos, training_heldout_mask, save_dir=save_dir)
        _merge_logs_into_logger(metrics, new_metrics)
        if hasattr(env, 'n_states'): 
            assert reward.shape[0] == env.n_states, f"Reward shape mismatch: {reward.shape[0]} vs {env.n_states}"
        additional_data = {
            'training_logs': metrics,
            'reward': reward,
            'state_encoder': state_encoder,
            'policy': agent.policy,
            'discriminator': agent.discriminator,
            'action_dim': action_dim,
            **agent_data
        }
    elif method == 'causal_airl':
        state_encoder = create_cartpole_encoder() if isinstance(env, CartPoleWrapper) else create_gridworld_encoder(grid_size=env.grid_size[1])
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
            invariance_penalty=cfg['irl']['inv_coeff'],
            lr=cfg['irl']['lr'],
            state_encoder=state_encoder,
            action_encoder=action_encoder,
            reward_logit_clip=cfg['irl'].get('reward_logit_clip')
        )
        reward, new_metrics, agent_data = agent.train(cfg, env, demos, training_heldout_mask, save_dir=save_dir)
        _merge_logs_into_logger(metrics, new_metrics)
        if hasattr(env, 'n_states'): 
            assert reward.shape[0] == env.n_states, f"Reward shape mismatch: {reward.shape[0]} vs {env.n_states}"
        additional_data = {
            'training_logs': metrics,
            'agent': agent,  # Store agent for later use
            'state_encoder': state_encoder,
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

    # Final metrics summary (append last values for key series)
    try:
        logs = metrics.get_logs() if hasattr(metrics, "get_logs") else metrics
        def last(key):
            v = logs.get(key, None)
            if isinstance(v, list) and v: return v[-1]
            return v
        final_metrics = {}
        for k in ["wall_time_sec","env_steps","reward_correlation","value_correlation",
                  "policy_agreement","avg_episode_return","avg_episode_length",
                  "episode_return_std"]:
            v = last(k)
            if v is not None: final_metrics[f"final_{k}"] = float(v) if isinstance(v, (int,float)) else v
        if final_metrics:
            metrics.log(final_metrics)
            print(f"[SUMMARY] Final metrics: {final_metrics}")
    except Exception as _e:
        print(f"[WARN] Failed to write final summary: {_e}")

    # Log minimal, consistent env meta into the metrics for traceability
    try:
        metrics.log({
            "env_confounded": bool(cfg.get('env', {}).get('confounded', False)) or hasattr(env, 'confounder_values'),
            "env_slip_prob": float(cfg.get('env', {}).get('slip_prob', 0.0)),
            "eval_test_z": cfg.get('eval', {}).get('test_z', None)
        })
    except Exception as _e:
        print(f"[WARN] Failed to log env meta: {_e}")

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
                # Additional direct evaluation with current policy (rollouts)
                try:
                    policy = additional_data.get('policy')
                    if policy is not None:
                        R, L = [], []
                        episodes = 10
                        for _ in range(episodes):
                            obs = env.reset()
                            if isinstance(obs, tuple):
                                obs = obs[0]
                            done = False
                            steps = 0
                            total_return = 0.0
                            # Ensure rollout matches the training input pipeline
                            _cp_enc = additional_data.get('state_encoder', None)
                            if _cp_enc is None:
                                _cp_enc = create_cartpole_encoder()
                            while not done and steps < 500:
                                with torch.no_grad():
                                    _x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                                    _x_enc = _cp_enc(_x)
                                    action = policy(_x_enc).sample().item()
                                next_obs, r, terminated, truncated, _ = env.step(action)
                                done = terminated or truncated
                                total_return += float(r)
                                obs = next_obs
                                steps += 1
                            R.append(total_return); L.append(steps)
                        max_steps = getattr(getattr(env, 'env', None), 'spec', None).max_episode_steps if hasattr(getattr(env, 'env', None), 'spec') and getattr(env.env, 'spec') else 500
                        success_thresh = max_steps - 5 if isinstance(max_steps, int) else 495
                        success_rate = float(np.mean(np.array(L) >= success_thresh))
                        metrics.log({
                            'eval_return_mean': float(np.mean(R)),
                            'eval_return_std': float(np.std(R)),
                            'eval_length_mean': float(np.mean(L)),
                            'success_rate': success_rate
                        })
                except Exception as _e:
                    print(f"[WARN] CartPole rollout eval failed: {_e}")

        else:
            # Pass precomputed T if available
            learned_state_reward = np.asarray(reward, dtype=float).reshape(-1)
            # Use potential-stripped reward for AIRL on GridWorld to get meaningful r_true vs r_learned correlations
            if method == 'airl' and hasattr(env, 'grid_size'):
                try:
                    se = additional_data.get('state_encoder')
                    if se is None:
                        # Fallback: construct the correct encoder for GridWorld
                        se = create_gridworld_encoder(grid_size=env.grid_size[1])
                    learned_state_reward = _compute_state_reward_from_airl(
                        additional_data['discriminator'], se, env
                    )
                    np.save(os.path.join(save_dir, 'learned_reward_state.npy'), learned_state_reward)
                except Exception as _e:
                    print(f"[WARN] Failed to compute state-only reward for AIRL: {_e}")

            print(f"[DEBUG] Learned reward vector shape: {learned_state_reward.shape}")
            T = additional_data.get('T', None)
            if T is not None:
                print(f"[{method.upper()}] Using {'sparse' if issparse(T) else 'dense'} transition matrix")

            # Persist transition matrix for reproducibility
            try:
                if issparse(T):
                    save_npz(os.path.join(save_dir, 'T_sparse.npz'), T)
                    additional_data['transition_matrix_file'] = 'T_sparse.npz'
                else:
                    np.save(os.path.join(save_dir, 'T.npy'), T)
                    additional_data['transition_matrix_file'] = 'T.npy'
            except Exception as e:
                print(f'[WARN] Persist T failed: {e}')

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

            # Enforce test_z for cross-confounder evaluation
            test_z = cfg['eval'].get('test_z', None)
            if test_z is not None and hasattr(env, 'set_confounder'):
                print(f"[Info] Setting environment confounder to test_z={test_z} for evaluation")
                if isinstance(env, CartPoleWrapper):
                    env.set_confounder(float(test_z))
                else:
                    env.z = test_z  # For ConfoundedGridWorld

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
                    # Approximate V from training logs, avoid using undefined eval_results
                    logs = metrics.get_logs() if hasattr(metrics, 'get_logs') else {}
                    avg_ret_list = logs.get('avg_episode_return', [])
                    avg_ret = float(np.mean(avg_ret_list)) if len(avg_ret_list) > 0 else 0.0
                    V_approx = np.array([avg_ret])

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

    # Configurable eval horizon (default 1000 for backward-compat)
    eval_max_steps = _infer_eval_max_steps(env, cfg)

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
                state_encoder = create_gridworld_encoder(grid_size=env.grid_size[1])
            else:
                raise ValueError("Unknown environment type for state encoder.")

            learned_trajectories = save_trajectories(
                policy, env, n_traj=10, device=device, state_encoder=state_encoder, max_steps=eval_max_steps
            )
            np.save(os.path.join(save_dir, 'trajectories.npy'), np.array(learned_trajectories, dtype=object))
            
            # Generate per-Z trajectories for confounded environments
            if hasattr(env, 'confounder_values'):
                for z in env.confounder_values:
                    z_trajs = save_trajectories(
                        policy, env, n_traj=10, z=z, device=device, state_encoder=state_encoder, max_steps=eval_max_steps
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
                policy_fn, env, n_traj=10, torch_policy=False, max_steps=eval_max_steps
            )
            np.save(os.path.join(save_dir, 'trajectories.npy'), np.array(learned_trajectories, dtype=object))
            
            # Generate per-Z trajectories for confounded environments
            if hasattr(env, 'confounder_values'):
                for z in env.confounder_values:
                    z_trajs = save_trajectories(
                        policy_fn, env, n_traj=10, z=z, torch_policy=False, max_steps=eval_max_steps
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
    # Trajectory diversity + per-episode eval metrics (guarded)
    if learned_trajectories is not None:
        try:
            visits = {}
            # Per-episode statistics
            ep_lengths = []
            ep_terminated = []
            for traj in learned_trajectories:
                # episode length: if terminal sentinel (last action None), count steps before it
                if len(traj) > 0 and traj[-1][1] is None:
                    ep_len = max(0, len(traj) - 1)
                    iter_pairs = traj[:-1]
                    ep_terminated.append(True)
                else:
                    ep_len = len(traj)
                    iter_pairs = traj
                    ep_terminated.append(False)
                ep_lengths.append(ep_len)
                for (s, a) in iter_pairs:  # Exclude final state
                    s_key = tuple(s) if hasattr(s, '__iter__') else s
                    visits[s_key] = visits.get(s_key, 0) + 1

            if visits:
                counts = np.array(list(visits.values()), dtype=float)
                p = counts / counts.sum()
                traj_entropy = float(-np.sum(p * np.log(p + 1e-12)))
                new_metrics["trajectory_entropy"] = traj_entropy
            else:
                new_metrics["trajectory_entropy"] = 0.0

            # Per-episode eval metrics
            if ep_lengths:
                ep_lengths = np.asarray(ep_lengths, dtype=float)
                ep_terminated = np.asarray(ep_terminated, dtype=bool)
                new_metrics["eval_episode_length_mean"] = float(np.mean(ep_lengths))
                if np.any(ep_terminated):
                    new_metrics["eval_steps_to_goal_mean"] = float(np.mean(ep_lengths[ep_terminated]))
                else:
                    new_metrics["eval_steps_to_goal_mean"] = None
                new_metrics["eval_success_rate"] = float(np.mean(ep_terminated))
                new_metrics["eval_timeout_rate"] = float(1.0 - new_metrics["eval_success_rate"])

        except Exception as e:
            print(f"[WARN] Trajectory diversity computation failed: {e}")
            new_metrics["trajectory_entropy"] = None
    else:
        new_metrics["trajectory_entropy"] = None

    # Reward distribution analysis (fixed)
    if reward is not None and hasattr(env, 'grid_size'):
        rw = np.asarray(reward).ravel().astype(float)
        reward_stats = {
            'reward_sparsity': float((np.abs(rw) < 1e-6).mean()),
            'reward_range': float(rw.max() - rw.min()) if rw.size > 0 else 0.0,
            'reward_std': float(rw.std()) if rw.size > 0 else 0.0,
            'reward_skewness': float(((rw - rw.mean())**3).mean() / (rw.std()**3 + 1e-8)) # Reward skewness (third standardized moment)
        }

        # Gini coefficient on |reward|
        absr = np.abs(rw)
        if absr.sum() > 0 and absr.size > 1:
            s = np.sort(absr)
            n = s.size
            gini = (2*np.arange(1,n+1)-n-1).dot(s)/(n*s.sum())
            reward_stats['reward_gini_abs'] = float(gini)

        # Histogram entropy on normalized |reward|
        if absr.max() > absr.min():
            hist, _ = np.histogram(absr, bins=min(20, len(absr)//5+1), density=True)
            hist = hist[hist > 0]  # Remove zero bins
            if len(hist) > 1:
                p_hist = hist / hist.sum()
                hist_entropy = float(-np.sum(p_hist * np.log(p_hist)))
                reward_stats['reward_hist_entropy'] = hist_entropy

        new_metrics.update(reward_stats)

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
