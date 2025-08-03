import os
import subprocess
import itertools
import yaml
import json
import csv
import argparse
from copy import deepcopy

def sweep_grid(base_config, sweep_params, save_dir_root):
    """Run parameter sweeps for all supported methods"""
    keys = list(sweep_params.keys())
    values = list(sweep_params.values())
    
    for combo in itertools.product(*values):
        config = deepcopy(base_config)
        name_parts = []
        
        # Update config with sweep values
        for k, v in zip(keys, combo):
            # Handle nested keys (e.g., "irl.method")
            key_parts = k.split('.')
            current = config
            for part in key_parts[:-1]:
                current = current.setdefault(part, {})
            current[key_parts[-1]] = yaml.safe_load(v)
            name_parts.append(f"{key_parts[-1]}-{v}")
        
        # Create run directory
        run_name = "_".join(name_parts)
        run_dir = os.path.join(save_dir_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        # Save config and run experiment
        config_path = os.path.join(run_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Run experiment
        subprocess.run(["python", "-m", "experiments.run_experiment", "--config", config_path])
        
        # Add to summary CSV
        try:
            # Load results
            config_path = os.path.join(run_dir, 'config.yaml')
            metrics_path = os.path.join(run_dir, 'metrics.json')
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Prepare CSV row
            row = {
                'method': config['irl']['method'],
                'env': config['env']['name'],
                'train_z': config['expert'].get('confounder_value', None),
                'test_z': config['eval'].get('test_z', None),
                'reward_mse': metrics.get('reward_mse'),
                'trajectory_overlap': metrics.get('trajectory_overlap'),
                'reward_variance': metrics.get('reward_variance')
            }
            
            # Append to summary
            csv_path = os.path.join(save_dir_root, 'generalization_summary.csv')
            write_header = not os.path.exists(csv_path)
            
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

        except Exception as e:
            print(f"Error logging sweep results: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results/sweeps")
    parser.add_argument("--sweep_params", type=str, required=False)
    parser.add_argument("--values", type=str, required=False)
    args = parser.parse_args()

    # Load base config
    with open(args.base_config, "r") as f:
        base_config = yaml.safe_load(f)

    # Parse custom sweep params if provided via CLI
    if args.sweep_params and args.values:
        keys = args.sweep_params.split(":")
        values = [v.split(",") for v in args.values.split(":")]
        sweep_params = dict(zip(keys, values))
    else:
        # Sweep configuration
        sweep_params = {
            "irl.method": ["airl", "causal_airl"],
            "irl.gamma": [0.9, 0.95, 0.99],
            "expert.num_trajectories": [5, 10, 20, 50],
            "env.slip_prob": [0.0, 0.1, 0.2],
            "env.reward_type": ["sparse", "shaped"]
        }

    sweep_grid(base_config, sweep_params, args.save_dir)
