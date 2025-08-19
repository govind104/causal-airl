import json
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def summarize_experiments(root):
    """Aggregate results dynamically from nested experiment run directories"""

    runs = []
    root_path = Path(root)

    # Recursively find all directories that contain both metrics.json and config.json
    # This handles both flat structure and nested timestamped directories
    def find_experiment_dirs(path):
        experiment_dirs = []
        try:
            if path.is_dir():
                # Check if this directory itself has the required files
                if (path / "metrics.json").exists() and (path / "config.json").exists():
                    experiment_dirs.append(path)

                # Recursively check subdirectories
                for item in path.iterdir():
                    if item.is_dir():
                        experiment_dirs.extend(find_experiment_dirs(item))
        except PermissionError:
            print(f"[skip] Permission denied: {path}")
        return experiment_dirs

    # Find all experiment directories
    experiment_dirs = find_experiment_dirs(root_path)

    # Process each experiment directory
    for d in experiment_dirs:
        m = d / "metrics.json"
        c = d / "config.json"

        try:
            metrics = json.loads(m.read_text())
            config = json.loads(c.read_text())

            info = {
                "run_dir": d.relative_to(root_path).as_posix(),
                "method": config.get("irl", {}).get("method"),
                "env_name": config.get("env", {}).get("name"),
                "confounded": config.get("env", {}).get("confounded", False),
                "heldout_region": config.get("eval", {}).get("heldout_region"),
                "seed": config.get("train", {}).get("seed"),
                "save_dir": config.get("eval", {}).get("save_dir"),  # Track original save_dir
            }

            # Flatten last values for list-logs
            for k, v in metrics.items():
                if isinstance(v, list) and v:
                    info[f"final_{k}"] = v[-1]
                elif not isinstance(v, list):
                    info[k] = v

            runs.append(info)

        except Exception as e:
            print(f"[skip] {d}: {e}")

    if not runs:
        print("No runs found")
        return

    # Ensure per-episode eval metrics columns are present (NaN where missing)
    eval_cols = [
        "eval_success_rate",
        "eval_timeout_rate",
        "eval_steps_to_goal_mean",
        "eval_episode_length_mean",
    ]

    # Save detailed results
    df = pd.DataFrame(runs)

    # Add missing eval columns as float NaNs (avoid pd.NA which forces object dtype)
    for col in eval_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Coverage log
    k_present = int(df[eval_cols].notna().any(axis=1).sum())
    n_total = int(len(df))
    print(f"[summarize] per-episode metrics present in {k_present}/{n_total} runs")

    # Coerce numeric candidates to floats; non-numeric → NaN
    numeric_candidates = [c for c in df.columns if c.startswith("final_")] + eval_cols
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df.to_csv(root_path / "detailed_results.csv", index=False)
    print(f"Saved detailed results: {root_path / 'detailed_results.csv'}")

    # Create summary by grouping key variables
    candidate_keys = [
        "final_reward_correlation",
        "final_value_correlation",
        "final_policy_agreement",
        "final_wall_time_sec",
        "final_env_steps",
        "final_avg_episode_return",
        # include eval per-episode metrics when present
        "eval_success_rate",
        "eval_timeout_rate",
        "eval_steps_to_goal_mean",
        "eval_episode_length_mean",
    ]
    # Keep only columns that exist and have at least one numeric (non-NaN) value
    keys = []
    for k in candidate_keys:
        if k in df.columns:
            s = pd.to_numeric(df[k], errors="coerce")
            if s.notna().any():
                keys.append(k)

    group = [k for k in ["method", "env_name", "confounded", "heldout_region"]
            if k in df.columns]

    if group and keys:
        summary = (df.groupby(group)[keys]
                      .agg(["mean", "std", "count"])
                      .round(4))
        summary.to_csv(root_path / "experiment_summary.csv")
        print(f"Saved summary: {root_path / 'experiment_summary.csv'}")

    print(f"\nFound {len(runs)} runs across {len(df.groupby(group)) if group else 1} configurations")

    # Print structure summary for debugging
    unique_save_dirs = df['save_dir'].unique()
    print(f"Save directories found: {list(unique_save_dirs)}")


def summarize_multiple_roots(roots):
    """Aggregate results from multiple root directories"""

    all_runs = []

    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"[skip] Directory does not exist: {root}")
            continue

        print(f"\nProcessing {root}...")

        # Use the same logic as summarize_experiments but collect runs
        def find_experiment_dirs(path):
            experiment_dirs = []
            try:
                if path.is_dir():
                    if (path / "metrics.json").exists() and (path / "config.json").exists():
                        experiment_dirs.append(path)

                    for item in path.iterdir():
                        if item.is_dir():
                            experiment_dirs.extend(find_experiment_dirs(item))
            except PermissionError:
                print(f"[skip] Permission denied: {path}")
            return experiment_dirs

        experiment_dirs = find_experiment_dirs(root_path)

        for d in experiment_dirs:
            m = d / "metrics.json"
            c = d / "config.json"

            try:
                metrics = json.loads(m.read_text())
                config = json.loads(c.read_text())

                info = {
                    "root_dir": root,
                    "run_dir": d.relative_to(root_path).as_posix(),
                    "method": config.get("irl", {}).get("method"),
                    "env_name": config.get("env", {}).get("name"),
                    "confounded": config.get("env", {}).get("confounded", False),
                    "heldout_region": config.get("eval", {}).get("heldout_region"),
                    "seed": config.get("train", {}).get("seed"),
                    "save_dir": config.get("eval", {}).get("save_dir"),
                }

                for k, v in metrics.items():
                    if isinstance(v, list) and v:
                        info[f"final_{k}"] = v[-1]
                    elif not isinstance(v, list):
                        info[k] = v

                all_runs.append(info)

            except Exception as e:
                print(f"[skip] {d}: {e}")

    if not all_runs:
        print("No runs found across all directories")
        return

    # Save combined results
    df = pd.DataFrame(all_runs)

    # Ensure eval columns exist and log coverage across roots
    eval_cols = ["eval_success_rate","eval_timeout_rate","eval_steps_to_goal_mean","eval_episode_length_mean"]
    for col in eval_cols:
        if col not in df.columns:
            df[col] = np.nan
    # Coerce numeric candidates to floats
    numeric_candidates = [c for c in df.columns if c.startswith("final_")] + eval_cols
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"[summarize] per-episode metrics present in {int(df[eval_cols].notna().any(axis=1).sum())}/{len(df)} runs (across roots)")

    output_path = Path("combined_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved combined results: {output_path} (rows={len(df)})")

    print(f"Found {len(all_runs)} total runs across {len(set(df['root_dir']))} directories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize experiment results.")
    parser.add_argument("--roots", nargs="+", default=None,
                        help="Root directories containing run folders (can pass multiple).")
    # Back-compat: also accept positional roots; merge with --roots.
    parser.add_argument("positional_roots", nargs="*", help="(Optional) roots as positional args.")
    args = parser.parse_args()

    # Build effective roots: merge flag + positional, dedup (order-preserving).
    roots = []
    if args.roots:
        roots.extend(args.roots)
    if args.positional_roots:
        roots.extend(args.positional_roots)
    if not roots:
        roots = ["results/validation"]
    # Dedup while preserving order
    seen = set()
    effective_roots = []
    for r in roots:
        if r not in seen:
            effective_roots.append(r); seen.add(r)

    # Quick discovered-run count (first-line log)
    def _count_runs(root: str) -> int:
        root_path = Path(root)
        cnt = 0
        def walk(p: Path):
            nonlocal cnt
            try:
                if p.is_dir():
                    if (p / "metrics.json").exists() and (p / "config.json").exists():
                        cnt += 1
                    for it in p.iterdir():
                        if it.is_dir():
                            walk(it)
            except PermissionError:
                print(f"[skip] Permission denied: {p}")
        if root_path.exists():
            walk(root_path)
        else:
            print(f"[skip] Directory does not exist: {root}")
        return cnt

    discovered = sum(_count_runs(r) for r in effective_roots)
    print(f"[summarize] roots={effective_roots} | discovered_runs={discovered}")

    # Route: single vs multi-root (by effective list length; continue even if some are missing)
    if len(effective_roots) == 1:
        summarize_experiments(effective_roots[0])
    else:
        summarize_multiple_roots(effective_roots)
