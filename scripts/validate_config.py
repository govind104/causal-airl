import yaml
import sys
import os

REQUIRED_KEYS = {
    'env': ['name'],
    'expert': ['num_trajectories', 'optimality'],
    'irl': ['method', 'gamma'],
    'train': ['seed'],
    'eval': ['save_dir']
}

def validate_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    errors = []
    for section, keys in REQUIRED_KEYS.items():
        if section not in cfg:
            errors.append(f"Missing section: {section}")
            continue
            
        for key in keys:
            if key not in cfg[section]:
                errors.append(f"Missing key: {section}.{key}")
    
    if errors:
        print(f"Invalid config {config_path}:")
        for error in errors:
            print(f"  - {error}")
        return False
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_config.py <config_path>")
        exit(1)
    
    valid = validate_config(sys.argv[1])
    exit(0 if valid else 1)