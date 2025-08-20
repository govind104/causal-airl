import json
import os
from typing import Dict, Any, Optional, List


def flatten_config(nested: dict, sep=".", prefix="") -> dict:
    """Recursively flatten nested dict to dot-separated keys.

    Args:
        nested: Nested dictionary to flatten
        sep: Separator for keys (default: ".")
        prefix: Prefix for current level keys

    Returns:
        Flattened dictionary with dot-separated keys
    """
    flat = {}
    for k, v in nested.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_config(v, sep=sep, prefix=key))
        else:
            flat[key] = v
    return flat

def get(d: dict, key: str, default=None, sep: str = "."):
    """Traverse dot-separated keys in dictionary.

    Args:
        d: Dictionary to traverse
        key: Dot-separated key string (e.g., "irl.method")
        default: Default value if key not found
        sep: Key separator (default: ".")

    Returns:
        Value at key path, or default if not found
    """
    # exact hit first (supports flattened dicts with dotted keys)
    if key in d:
        return d[key]
    if sep not in key:
        return d.get(key, default)

    keys = key.split(sep)
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def find_run_dirs(roots: List[str]) -> List[str]:
    """Find all leaf run directories containing metrics.json from multiple roots.

    Args:
        roots: List of root directories to search

    Returns:
        Sorted list of unique run directory paths
    """
    run_dirs = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Filter out hidden dirs in-place
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
            if 'metrics.json' in filenames:
                # Check if leaf - if no subdirs also contain metrics.json
                has_sub_runs = False
                for subdir in dirnames:
                    sub_path = os.path.join(dirpath, subdir)
                    if os.path.isfile(os.path.join(sub_path, 'metrics.json')):
                        has_sub_runs = True
                        break
                if not has_sub_runs:
                    run_dirs.add(dirpath)
    return sorted(run_dirs)

def load_config_with_fallback(run_dir: str) -> Dict[str, Any]:
    """Load config, preferring flattened version if available.

    Args:
        run_dir: Path to run directory

    Returns:
        Configuration dictionary (flattened if config_flat.json exists)
    """
    flat_path = os.path.join(run_dir, "config_flat.json")
    if os.path.exists(flat_path):
        with open(flat_path, "r") as f:
            return json.load(f)

    # Fallback to nested config and flatten in-memory
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            nested = json.load(f)
        return flatten_config(nested)

    return {}
