import argparse
import itertools
import os
import json
import time

from experiments.run_experiment import run_experiment, load_config, update_config_with_overrides, set_seed

def parse_grid(items):
    grid = {}
    for kv in items:
        k, vs = kv.split("=")
        grid[k] = vs.split(",")
    return grid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="base config path")
    parser.add_argument("--save_root", type=str, required=True, help="root to store runs")
    parser.add_argument("--grid", action="append", default=[], help="k=v1,v2,... overrides grid")
    parser.add_argument("--seed", type=int, default=42, help="global seed for sweep reproducibility")
    args = parser.parse_args()

    # Set global seed for entire sweep
    set_seed(args.seed)

    base = load_config(args.base)
    base.setdefault('eval', {})['save_dir'] = args.save_root
    base['_overrides'] = {}

    grid = parse_grid(args.grid)
    keys = sorted(grid.keys())
    total_experiments = len(list(itertools.product(*[grid[k] for k in keys])))

    failed_experiments = []

    for i, values in enumerate(itertools.product(*[grid[k] for k in keys])):
        overrides = [f"{k}={v}" for k, v in zip(keys, values)]
        cfg = update_config_with_overrides(dict(base), overrides)
        cfg['_overrides'] = {k: v for k, v in zip(keys, values)}

        print(f"Running experiment {i+1}/{total_experiments}: {overrides}")

        try:
            run_experiment(cfg)
            print(f"✓ Experiment {i+1} completed successfully")
        except Exception as e:
            error_msg = f"✗ Experiment {i+1} failed: {str(e)}"
            print(error_msg)
            failed_experiments.append((overrides, str(e)))
            continue  # Continue with next experiment

    # Summary report
    print(f"\nSweep completed: {total_experiments - len(failed_experiments)}/{total_experiments} succeeded")
    if failed_experiments:
        print(f"\nFailed experiments:")
        for overrides, error in failed_experiments:
            print(f"  {overrides}: {error}")


if __name__ == "__main__":
    main()
