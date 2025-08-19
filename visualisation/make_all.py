import argparse
import os
import sys
import subprocess
from typing import List, Dict

from visualisation.utils_config import find_run_dirs

DEFAULT_ROOTS = [
    "results/validation",
    "results/gridworld_baselines",
    "results/airl_ablation",
    "results/causal_airl_ablation",
    "results/confounded",
    "results/generalization",
]

def _dedup(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _out(path: str, sub: str) -> str:
    p = os.path.join(path, sub)
    os.makedirs(p, exist_ok=True)
    return p

def _build(cmd_mod: str, outdir: str, roots: List[str], extra: List[str]) -> List[str]:
    return [sys.executable, "-m", f"visualisation.{cmd_mod}", "--roots", *roots, "--out", outdir, *extra]

def _run(cmd: List[str], dry: bool) -> Dict:
    if dry:
        print("DRY-RUN:", " ".join(cmd))
        return {"returncode": 0, "stdout": b"", "stderr": b"", "skipped": True}
    try:
        res = subprocess.run(cmd, check=False, capture_output=True)
        return {"returncode": res.returncode, "stdout": res.stdout, "stderr": res.stderr, "skipped": False}
    except Exception as e:
        return {"returncode": 127, "stdout": b"", "stderr": str(e).encode("utf-8"), "skipped": False}

def _pick_out_subdir(mod: str) -> str:
    # Conventional subfolders for each script
    mapping = {
        "plot_ablation_summaries": "summary",
        "plot_training_curves": "training",
        "plot_rewards": "heatmaps",
        "plot_reward_differences": "diff",
        "plot_causal_invariance": "invariance",
        "plot_heldout_overlay": "heldout",
        "plot_policy_fields": "policy",
        "plot_tradeoffs": "tradeoffs",
        "plot_checkpoint_evolution": "checkpoints",
        "plot_crossz_bars": "crossz",
        "plot_reward_stats": "reward_stats",
        "plot_trajectory_diversity": "rollouts",
        "plot_invariance_violations": "invariance",
        "plot_summary_grid": "panels",
    }
    return mapping.get(mod, mod)

def _steps(out_base: str, roots: List[str]) -> List[Dict]:
    """
    Ordered steps with tags and sensible defaults per script.
    """
    py = sys.executable  # not used directly; kept for clarity
    s = []
    # 1
    s.append(dict(
        mod="plot_ablation_summaries",
        tags=["ablations"],
        cmd=lambda: _build(
            "plot_ablation_summaries",
            _out(out_base, _pick_out_subdir("plot_ablation_summaries")),
            roots,
            [
                "--csv", "results/validation/experiment_summary.csv",
                "--x", "expert.num_trajectories",
                "--metrics", "final_reward_correlation", "final_policy_agreement",
                "--groupby", "scenario", "method",
            ],
        ),
    ))
    # 2
    s.append(dict(
        mod="plot_training_curves",
        tags=["training"],
        cmd=lambda: _build("plot_training_curves", _out(out_base, _pick_out_subdir("plot_training_curves")), roots,
                           ["--logfile", "training_logs.json"]),
    ))
    # 3
    s.append(dict(
        mod="plot_rewards",
        tags=["rewards"],
        cmd=lambda: _build("plot_rewards", _out(out_base, _pick_out_subdir("plot_rewards")), roots, []),
    ))
    # 4
    s.append(dict(
        mod="plot_reward_differences",
        tags=["rewards"],
        cmd=lambda: _build("plot_reward_differences", _out(out_base, _pick_out_subdir("plot_reward_differences")), roots, []),
    ))
    # 5
    s.append(dict(
        mod="plot_causal_invariance",
        tags=["invariance"],
        cmd=lambda: _build(
            "plot_causal_invariance",
            _out(out_base, _pick_out_subdir("plot_causal_invariance")),
            roots,
            ["--mode", "across_runs", "--groupby", "env.confounder_value", "--metric", "var"],
        ),
    ))
    # 6
    s.append(dict(
        mod="plot_heldout_overlay",
        tags=["heldout"],
        cmd=lambda: _build("plot_heldout_overlay", _out(out_base, _pick_out_subdir("plot_heldout_overlay")), roots, []),
    ))
    # 7
    s.append(dict(
        mod="plot_policy_fields",
        tags=["policy"],
        cmd=lambda: _build("plot_policy_fields", _out(out_base, _pick_out_subdir("plot_policy_fields")), roots,
                           ["--show_tick_labels"]),
    ))
    # 8
    s.append(dict(
        mod="plot_tradeoffs",
        tags=["tradeoffs", "rewards"],
        cmd=lambda: _build(
            "plot_tradeoffs",
            _out(out_base, _pick_out_subdir("plot_tradeoffs")),
            roots,
            [
                "--metric", "final_reward_correlation",
                "--x", "final_env_steps",
                "--facet", "scenario",
                "--facet_rows",
            ],
        ),
    ))
    # 9
    s.append(dict(
        mod="plot_checkpoint_evolution",
        tags=["checkpoints"],
        cmd=lambda: _build("plot_checkpoint_evolution", _out(out_base, _pick_out_subdir("plot_checkpoint_evolution")), roots, []),
    ))
    # 10
    s.append(dict(
        mod="plot_crossz_bars",
        tags=["crossz"],
        cmd=lambda: _build("plot_crossz_bars", _out(out_base, _pick_out_subdir("plot_crossz_bars")), roots, []),
    ))
    # 11
    s.append(dict(
        mod="plot_reward_stats",
        tags=["rewards"],
        cmd=lambda: _build("plot_reward_stats", _out(out_base, _pick_out_subdir("plot_reward_stats")), roots, []),
    ))
    # 12
    s.append(dict(
        mod="plot_trajectory_diversity",
        tags=["rollouts"],
        cmd=lambda: _build("plot_trajectory_diversity", _out(out_base, _pick_out_subdir("plot_trajectory_diversity")), roots, []),
    ))
    # 13
    s.append(dict(
        mod="plot_invariance_violations",
        tags=["invariance"],
        cmd=lambda: _build(
            "plot_invariance_violations",
            _out(out_base, _pick_out_subdir("plot_invariance_violations")),
            roots,
            ["--K", "10"],
        ),
    ))
    return s

def _filter_steps(steps: List[Dict], only: List[str], skip: List[str]) -> List[Dict]:
    if only:
        only_set = set([x.strip() for x in ",".join(only).split(",") if x.strip()])
        steps = [st for st in steps if (st["mod"] in only_set) or (set(st["tags"]) & only_set)]
    if skip:
        skip_set = set([x.strip() for x in ",".join(skip).split(",") if x.strip()])
        steps = [st for st in steps if (st["mod"] not in skip_set) and not (set(st["tags"]) & skip_set)]
    return steps

def main():
    p = argparse.ArgumentParser(description="Make all visualisations (orchestrator).")
    p.add_argument("--roots", nargs="+", default=[], help="Root directories with runs")
    p.add_argument("--out", default="results/figures", help="Base output directory (default: results/figures)")
    p.add_argument("--only", nargs="*", default=None, help="Subset by tags or script names, comma-separated")
    p.add_argument("--skip", nargs="*", default=None, help="Skip tags or script names, comma-separated")
    p.add_argument("--dry_run", action="store_true", help="Print commands only, do not execute")
    p.add_argument("--seed", type=int, default=0, help="Seed (passed where applicable)")
    args = p.parse_args()

    eff_roots = _dedup(list(args.roots) + DEFAULT_ROOTS)
    discovered = find_run_dirs(eff_roots)
    print(f"[make_all] roots={eff_roots} | discovered_runs={len(discovered)}")

    steps = _steps(args.out, eff_roots)
    steps = _filter_steps(steps, args.only or [], args.skip or [])
    print(f"[make_all] selected_steps={ [st['mod'] for st in steps] }")
    if args.dry_run and not steps:
        print("[make_all] nothing to do (dry run)")
        return

    ok = 0
    fail = 0
    skipped = 0

    for st in steps:
        cmd = st["cmd"]()
        # pass seed where the target script supports it (safe subset)
        if st["mod"] in ("plot_summary_grid",):
            cmd += ["--seed", str(args.seed)]
        res = _run(cmd, args.dry_run)
        rc = res["returncode"]
        if res["skipped"]:
            print(f"• {st['mod']}: DRY-RUN")
            skipped += 1
            continue
        if rc == 0:
            print(f"✓ {st['mod']}")
            ok += 1
        else:
            # print a single-line reason
            reason = (res["stderr"] or res["stdout"] or b"").decode("utf-8", errors="ignore").strip()
            if reason:
                # shorten to last non-empty line
                lines = [ln for ln in reason.splitlines() if ln.strip()]
                reason = lines[-1] if lines else reason
            print(f"✗ {st['mod']} — {reason}")
            fail += 1

    print(f"[make_all] done: ok={ok}, failed={fail}, dry_skipped={skipped} | out_base={args.out}")
    # Exit 0 if at least one success; else 1
    sys.exit(0 if ok > 0 else 1)

if __name__ == "__main__":
    main()
