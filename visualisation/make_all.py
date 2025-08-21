import argparse
import os
import sys
import subprocess
from typing import List, Dict

from visualisation.utils_config import find_run_dirs


DEFAULT_ROOTS = [
    "results/gridworld_baselines",
    "results/confounded",
    "results/generalization",
    "results/generalisation",
    "results/airl_hparams",
    "results/causal_airl_hparams",
    "results/airl_scenarios",
    "results/causal_airl_scenarios",
    "results/airl_scenario_sweep",
    "results/causal_airl_scenario_sweep",
    "results/scaling",
]

# Fan-out parameter grids (MakeAllPlan.txt)

# Methods
METHODS_ALL = ['ng', 'maxent', 'airl', 'causal_airl']
METHODS_DEEP = ['airl', 'causal_airl']
METHODS_CLASSIC = ['ng', 'maxent']

# Scenarios (labels emitted by scenario.py; kept here for future filtering if desired)
SCENARIOS = ['baseline','shaped','noisy','noisy_shaped','confounded','confounded_crossZ','heldout','fewshot']

# Ablation grids
ABLS_X = ['expert.num_trajectories','env.slip_prob','env.reward_type','irl.gamma']
ABLS_Y = ['final_reward_spearman','final_value_correlation','final_policy_agreement']
ABLS_GROUPS = ['method']

# Curves & tradeoffs
CURVE_METRICS = ['reward_correlation','policy_agreement']
TRADEOFF_PAIRS = [('final_reward_spearman','final_policy_agreement'),
                  ('final_reward_spearman','final_value_correlation')]

# Training-curve metrics (small fan-out, includes losses when available)
TRAIN_CURVE_METRICS = ['reward_correlation','policy_agreement','discriminator_loss','policy_loss']

# K loops
K_VALUES = [10]

# Compute–accuracy metrics
COMPUTE_METRICS = ['reward_spearman','value_correlation','policy_agreement']

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
        "plot_scaling_perf": "scaling",
        "plot_compute_vs_accuracy": "scaling",
        "generate_summary_tables": "tables",
    }
    return mapping.get(mod, mod)

def _steps(out_base: str, roots: List[str]) -> List[Dict]:
    """
    Ordered steps with tags and sensible defaults per script.
    """
    py = sys.executable  # not used directly; kept for clarity
    s = []
    # 1 — plot_ablation_summaries (fan-out over x,y,group)
    for x in ABLS_X:
        for y in ABLS_Y:
            for group in ABLS_GROUPS:
                s.append(dict(
                    mod="plot_ablation_summaries",
                    tags=["ablations"],
                    cmd=lambda x=x, y=y, group=group: _build(
                        "plot_ablation_summaries",
                        _out(out_base, os.path.join(_pick_out_subdir("plot_ablation_summaries"),
                                                    f"x={x}", f"y={y}", f"group={group}")),
                        roots,
                        ["--csv", "results/validation/experiment_summary.csv",
                         "--x", x,
                         "--metrics", y,
                         "--groupby", group],
                    ),
                ))
                # Deep-only per-method panels (AIRL & Causal-AIRL)
                for m in METHODS_DEEP:
                    s.append(dict(
                        mod="plot_ablation_summaries",
                        tags=["ablations"],
                        cmd=lambda x=x, y=y, group=group, m=m: _build(
                            "plot_ablation_summaries",
                            _out(out_base, os.path.join(
                                _pick_out_subdir("plot_ablation_summaries"),
                                f"x={x}", f"y={y}", f"group={group}",
                                "mset=deep", f"method={m}"
                            )),
                            roots,
                            ["--csv", "results/validation/experiment_summary.csv",
                             "--x", x,
                             "--metrics", y,
                             "--groupby", group,
                             "--method_filter", m],
                        ),
                    ))
    # 2 — plot_training_curves (fan-out over key metrics)
    for metric in TRAIN_CURVE_METRICS:
        s.append(dict(
            mod="plot_training_curves",
            tags=["training"],
            cmd=lambda metric=metric: _build(
                "plot_training_curves",
                _out(out_base, os.path.join(_pick_out_subdir("plot_training_curves"), f"metric={metric}")),
                roots,
                ["--logfile", "training_logs.json", "--metrics", metric],
            ),
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
            ["--mode", "across_runs",
             "--groupby", "env.confounder_value",
             "--metric", "var",
             *(["--perz_csv", "results/tables/perz_results.csv"]
               if os.path.exists("results/tables/perz_results.csv") else [])
            ],
        ),
    ))
    # 6
    s.append(dict(
        mod="plot_heldout_overlay",
        tags=["heldout"],
        cmd=lambda: _build("plot_heldout_overlay", _out(out_base, _pick_out_subdir("plot_heldout_overlay")), roots, []),
    ))
    # 7 — plot_policy_fields (base)
    s.append(dict(
        mod="plot_policy_fields",
        tags=["policy"],
        cmd=lambda: _build("plot_policy_fields", _out(out_base, _pick_out_subdir("plot_policy_fields")), roots,
                           ["--show_tick_labels"]),
    ))
    # 7b — plot_policy_fields (overlay trajectories + grid)
    s.append(dict(
        mod="plot_policy_fields",
        tags=["policy"],
        cmd=lambda: _build(
            "plot_policy_fields",
            _out(out_base, os.path.join(_pick_out_subdir("plot_policy_fields"), "overlay_traj")),
            roots,
            ["--show_tick_labels", "--overlay_trajectories", "--show_grid", "--grid_alpha", "0.2"],
        ),
    ))
    # 8 — plot_tradeoffs
    # 8a — Keep original figure (metric vs env steps)
    s.append(dict(
        mod="plot_tradeoffs",
        tags=["tradeoffs", "rewards"],
        cmd=lambda: _build(
            "plot_tradeoffs",
            _out(out_base, _pick_out_subdir("plot_tradeoffs")),
            roots,
            ["--metric", "final_reward_spearman",
             "--x", "final_env_steps",
             "--facet", "scenario",
             "--facet_rows"],
        ),
    ))
    # 8b — Plus additional tradeoff pairs (fan-out)
    for x, y in TRADEOFF_PAIRS:
        s.append(dict(
            mod="plot_tradeoffs",
            tags=["tradeoffs", "rewards"],
            cmd=lambda x=x, y=y: _build(
                "plot_tradeoffs",
                _out(out_base, os.path.join(_pick_out_subdir("plot_tradeoffs"),
                                            f"x={x}", f"y={y}")),
                roots,
                ["--metric", y,
                 "--x", x,
                 "--facet", "scenario",
                 "--facet_rows"],
            ),
        ))
    # 8c — plot_tradeoffs (compute-aware outside scaling: wall-time vs accuracy)
    s.append(dict(
        mod="plot_tradeoffs",
        tags=["tradeoffs", "rewards"],
        cmd=lambda: _build(
            "plot_tradeoffs",
            _out(out_base, os.path.join(_pick_out_subdir("plot_tradeoffs"),
                                        "x=final_wall_time_sec", "y=final_reward_spearman")),
            roots,
            ["--metric", "final_reward_spearman",
             "--x", "final_wall_time_sec",
             "--facet", "scenario",
             "--facet_rows"],
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
        cmd=lambda: _build(
            "plot_crossz_bars",
            _out(out_base, _pick_out_subdir("plot_crossz_bars")),
            roots,
            [*(["--perz_csv", "results/tables/perz_results.csv"]
               if os.path.exists("results/tables/perz_results.csv") else [])],
        ),
    ))
    # 11
    s.append(dict(
        mod="plot_reward_stats",
        tags=["rewards"],
        cmd=lambda: _build(
            "plot_reward_stats",
            _out(out_base, _pick_out_subdir("plot_reward_stats")),
            roots,
            [],
        ),
    ))
    # 12
    s.append(dict(
        mod="plot_trajectory_diversity",
        tags=["rollouts"],
        cmd=lambda: _build("plot_trajectory_diversity", _out(out_base, _pick_out_subdir("plot_trajectory_diversity")), roots, []),
    ))
    # 13 — plot_invariance_violations (fan-out over K; defaults to [10])
    for K in K_VALUES:
        s.append(dict(
            mod="plot_invariance_violations",
            tags=["invariance"],
            cmd=lambda K=K: _build(
                "plot_invariance_violations",
                _out(out_base, os.path.join(_pick_out_subdir("plot_invariance_violations"),
                                            f"K={K}")),
                roots,
                ["--K", str(K)],
            ),
        ))
    # 14
    s.append(dict(
        mod="plot_scaling_perf",
        tags=["scaling", "compute"],
        cmd=lambda: [sys.executable, "-m", "visualisation.plot_scaling_perf",
                     "--csv", os.path.join("results", "scaling", "perf.csv"),
                     "--out", _out(out_base, _pick_out_subdir("plot_scaling_perf"))],
    ))
    # 15 — plot_compute_vs_accuracy (fan-out over metrics)
    for metric in COMPUTE_METRICS:
        s.append(dict(
            mod="plot_compute_vs_accuracy",
            tags=["scaling", "compute"],
            cmd=lambda metric=metric: [sys.executable, "-m", "visualisation.plot_compute_vs_accuracy",
                                       "--perf", os.path.join("results", "scaling", "perf.csv"),
                                       "--metric", metric,
                                       "--out", _out(out_base, os.path.join(_pick_out_subdir("plot_compute_vs_accuracy"),
                                                                            f"metric={metric}"))],
        ))
    # 16 — generate_summary_tables (guarded: skip if CSV is absent)
    def _tables_cmd():
        csv_path = os.path.join("results", "validation", "experiment_summary.csv")
        if not os.path.exists(csv_path):
            print(f"[make_all] skip generate_summary_tables — missing CSV: {csv_path}")
            # return a no-op that succeeds
            return [sys.executable, "-c", "print('skip generate_summary_tables: missing CSV')"]
        return [sys.executable, "-m", "visualisation.generate_summary_tables",
                "--csv", csv_path,
                "--out", os.path.join(_out(out_base, _pick_out_subdir("generate_summary_tables")), "summary.csv"),
                "--md", os.path.join(_out(out_base, _pick_out_subdir("generate_summary_tables")), "summary.md")]
    s.append(dict(
        mod="generate_summary_tables",
        tags=["tables"],
        cmd=_tables_cmd,
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
            print(f"{st['mod']}: DRY-RUN")
            skipped += 1
            continue
        if rc == 0:
            print(f"{st['mod']}")
            ok += 1
        else:
            # print a single-line reason
            reason = (res["stderr"] or res["stdout"] or b"").decode("utf-8", errors="ignore").strip()
            if reason:
                # shorten to last non-empty line
                lines = [ln for ln in reason.splitlines() if ln.strip()]
                reason = lines[-1] if lines else reason
            print(f"{st['mod']} — {reason}")
            fail += 1

    print(f"[make_all] done: ok={ok}, failed={fail}, dry_skipped={skipped} | out_base={args.out}")
    # Exit 0 if at least one success; else 1
    sys.exit(0 if ok > 0 else 1)

if __name__ == "__main__":
    main()
