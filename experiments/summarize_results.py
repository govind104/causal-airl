import argparse
import json
import math
import os
import re
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# -------------------------------
# Discovery & IO utilities
# -------------------------------

DEFAULT_ROOTS = [
    "results/gridworld_baselines",
    "results/airl_hparams",
    "results/causal_airl_hparams",
    "results/airl_scenarios",
    "results/causal_airl_scenarios",
    "results/confounded",
    "results/generalization",
    "results/scaling",  # read-only for perf join
]

EXPERIMENT_TYPE_MAP = {
    "gridworld_baselines": "baselines",
    "airl_hparams": "hparams",
    "causal_airl_hparams": "hparams",
    "airl_scenarios": "scenarios",
    "causal_airl_scenarios": "scenarios",
    "confounded": "confounded",
    "generalization": "generalization",
    "generalisation": "generalization",
    "scaling": "scaling",
}

METHOD_NORMALISATION = {
    None: None,
    "airl": "airl",
    "AIRL": "airl",
    "causal_airl": "causal_airl",
    "CAUSAL_AIRL": "causal_airl",
    "causal-airl": "causal_airl",
    "maxent": "maxent",
    "MAXENT": "maxent",
    "ng": "ng",
    "NG": "ng",
}

NUMERIC_SCENARIO_KEYS = [
    "irl.gamma",
    "irl.entropy_coef",
    "irl.grad_clip_norm",
    "irl.kl_coeff",
    "irl.inv_coeff",
    "irl.latent_dim",
    "irl.num_z_samples",
    "expert.num_trajectories",
    "expert.confounder_value",
    "env.slip_prob",
]

GROUPING_SIGNATURE = [
    "method",
    "env_name",
    "irl.gamma",
    "expert.num_trajectories",
    "env.slip_prob",
    "env.reward_type",
    "irl.entropy_coef",
    "irl.grad_clip_norm",
    "irl.kl_coeff",
    "irl.inv_coeff",
    "irl.latent_dim",
    "irl.num_z_samples",
    "expert.confounder_value",
    "eval.test_z",
    "eval.heldout_region",
    "size_N",
]

FINAL_METRIC_BASES = [
    # Rank/value fidelity
    "reward_spearman",
    "reward_correlation",
    "value_correlation",
    "value_difference",
    "policy_agreement",
    "value_correlation_weighted",
    "policy_agreement_weighted",
    # Robustness / invariance
    "reward_variance",           # across-Z variance if logged
    "spearman_worstZ",
    "valuecorr_worstZ",
    # Compute
    "wall_time_sec",
    "env_steps",
    # Episode stats
    "eval_success_rate",
    "eval_timeout_rate",
    "eval_steps_to_goal_mean",
    "eval_episode_length_mean",
    # Training snapshot (optional)
    "disc_total_loss",
    "policy_loss",
    # Policy / confounded and trajectory metrics
    "confounded_expert_agreement",
    "trajectory_overlap",
    # Reward map stats
    "reward_sparsity",
    "reward_range",
    "reward_std",
    "reward_skewness",
    "reward_gini_abs",
    "reward_hist_entropy",
]

PERZ_SIDECARE_PATTERNS = [
    r"metrics_z[=_\-]?(\d+)\.json",  # metrics_z=0.json, metrics_z0.json
    r"metrics_by_z\.json",           # nested dict by z
    r"per_z\.json",                  # generic
]


def _dedup(seq: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _is_run_dir(d: Path) -> bool:
    """Run dir: has metrics.json and any of {config_flat.json, config.json}."""
    return (d / "metrics.json").exists() and ((d / "config_flat.json").exists() or (d / "config.json").exists())


def _discover_run_dirs(root: Path) -> List[Path]:
    """Recursively find run dirs; skip scaling as 'normal' runs."""
    out = []
    if not root.exists():
        return out
    for p in root.rglob("metrics.json"):
        run_dir = p.parent
        # Skip scaling runs (we only ingest scaling/perf.csv via join)
        if "results/scaling" in str(run_dir).replace("\\", "/"):
            continue
        if _is_run_dir(run_dir):
            out.append(run_dir)
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_config(run_dir: Path) -> Dict[str, Any]:
    """Prefer config_flat.json; fallback to config.json."""
    flat = run_dir / "config_flat.json"
    if flat.exists():
        return _read_json(flat)
    return _read_json(run_dir / "config.json")


def _last_finite(x: Any) -> float:
    """Return last finite value (lists → last finite; scalars → finite) else NaN."""
    try:
        if isinstance(x, list):
            for v in reversed(x):
                if v is not None and np.isfinite(v):
                    return float(v)
            return np.nan
        return float(x) if (x is not None and np.isfinite(x)) else np.nan
    except Exception:
        return np.nan


def _extract_final(metrics: Dict[str, Any], base: str) -> float:
    """
    Robust final extraction:
      prefer *_test → *_eval → base → *_train;
      if already logged as 'final_*', use its last finite directly.
    Also try weighted variants if base missing (handled by caller by passing accordingly).
    """
    # If the metrics has 'final_base' already, just use it (and do not 'final_final_*').
    final_key = f"final_{base}"
    if final_key in metrics:
        return _last_finite(metrics.get(final_key))
    # Ordered fallbacks
    for k in (f"{base}_test", f"{base}_eval", base, f"{base}_train"):
        if k in metrics:
            v = _last_finite(metrics[k])
            if not np.isnan(v):
                return v
    # As a last resort, try a few common misnamings
    for k in (f"{base}_final", f"{base}"):  # already tried 'base', keep for completeness
        if k in metrics:
            v = _last_finite(metrics[k])
            if not np.isnan(v):
                return v
    return np.nan


def _parse_timestamp_from_dir(dirname: str) -> Optional[str]:
    """
    Best-effort timestamp parsing from directory names:
    matches patterns like YYYYMMDD-HHMMSS or YYYY-MM-DD_HH-MM-SS.
    Returns the matched string if found.
    """
    s = os.path.basename(dirname)
    pats = [
        r"(\d{8}[-_]\d{6})",
        r"(\d{4}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2})",
    ]
    for pat in pats:
        m = re.search(pat, s)
        if m:
            return m.group(1)
    return None


def _normalise_method(m: Any) -> Optional[str]:
    if m is None:
        return None
    return METHOD_NORMALISATION.get(str(m), str(m).lower())


def _size_N_from_grid_size(val: Any) -> Optional[int]:
    """Derive N for square grids; otherwise None."""
    try:
        if isinstance(val, int):
            return int(val)
        if isinstance(val, (list, tuple)) and len(val) == 2 and val[0] == val[1]:
            return int(val[0])
        if isinstance(val, str) and "x" in val.lower():
            a, b = re.split("[xX]", val)
            if int(a) == int(b):
                return int(a)
    except Exception:
        pass
    return None


# -------------------------------
# Per-Z sidecar ingestion
# -------------------------------

def _collect_perz_values(run_dir: Path, bases: Iterable[str]) -> Dict[int, Dict[str, float]]:
    """
    Collect per-Z values if sidecar files exist. Returns {z: {base: value}}.
    Supports:
      - metrics_z=0.json / metrics_z0.json
      - metrics_by_z.json containing {z: {metric_key: series}}
    """
    out: Dict[int, Dict[str, float]] = {}
    # Pattern 1: explicit files per Z
    for sidecar in run_dir.glob("metrics_z*.json"):
        m = re.search(r"z[=_\-]?(\d+)", sidecar.name)
        if not m:
            continue
        z = int(m.group(1))
        data = _read_json(sidecar)
        out.setdefault(z, {})
        for base in bases:
            v = _extract_final(data, base)
            if not np.isnan(v):
                out[z][base] = v
    # Pattern 2: metrics_by_z.json
    mbz = run_dir / "metrics_by_z.json"
    if mbz.exists():
        data = _read_json(mbz)
        if isinstance(data, dict):
            for zk, sub in data.items():
                try:
                    z = int(zk)
                except Exception:
                    continue
                out.setdefault(z, {})
                if isinstance(sub, dict):
                    for base in bases:
                        # prefer *_test → *_eval → base → *_train within the sub-dict
                        v = None
                        for k in (f"{base}_test", f"{base}_eval", base, f"{base}_train", f"final_{base}"):
                            if k in sub:
                                v = _last_finite(sub[k])
                                if not np.isnan(v):
                                    break
                        if v is not None and not np.isnan(v):
                            out[z][base] = v
    return out


# -------------------------------
# Aggregation helpers
# -------------------------------

def _t_crit_95(df: int) -> float:
    """Two-sided 95% t critical; small lookup for df<=30; normal approx after."""
    # df: degrees of freedom
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
        18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    if df <= 0:
        return float("nan")
    if df <= 30:
        return table[df]
    return 1.960  # ~normal


def _mean_std_ci_str(vals: List[float]) -> Tuple[float, float, int, float, str]:
    xs = [float(v) for v in vals if (v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))))]
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"), 0, float("nan"), "NA")
    if n == 1:
        m = xs[0]
        return (m, 0.0, 1, 0.0, f"{m:.3f} ± 0.000")
    m = float(np.mean(xs))
    sd = float(np.std(xs, ddof=1))
    se = sd / math.sqrt(n)
    ci = _t_crit_95(n - 1) * se
    return (m, sd, n, ci, f"{m:.3f} ± {ci:.3f}")


# -------------------------------
# Collection & summarisation
# -------------------------------

def _collect_run_row(run_dir: Path, root: str) -> Tuple[Dict[str, Any], Dict[int, Dict[str, float]]]:
    metrics = _read_json(run_dir / "metrics.json")
    cfg = _load_config(run_dir)

    # Split flat or nested config into a uniform access
    def get(k: str, default=None):
        if k in cfg:
            return cfg.get(k, default)
        # dotted path from nested config.json
        node = cfg
        for part in k.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    # Identifiers
    env_name = get("env.name") or get("name") or get("env")
    method_raw = get("irl.method") or get("method")
    method = _normalise_method(method_raw)
    seed = get("train.seed") or get("seed")
    heldout_region = get("eval.heldout_region")
    test_z = get("eval.test_z")
    grid_size = get("env.grid_size")
    size_N = _size_N_from_grid_size(grid_size)
    timestamp = _parse_timestamp_from_dir(str(run_dir))
    # Experiment typing
    exp_root_name = os.path.basename(root.rstrip("/"))
    experiment_type = EXPERIMENT_TYPE_MAP.get(exp_root_name, exp_root_name)

    info: Dict[str, Any] = {
        "run_path": str(run_dir),
        "experiment_root": root,
        "experiment_type": experiment_type,
        "timestamp": timestamp,
        "method": method,
        "env_name": env_name,
        "train.seed": seed,
        # knobs
        "irl.gamma": get("irl.gamma"),
        "irl.entropy_coef": get("irl.entropy_coef"),
        "irl.grad_clip_norm": get("irl.grad_clip_norm"),
        "irl.kl_coeff": get("irl.kl_coeff"),
        "irl.inv_coeff": get("irl.inv_coeff"),
        "irl.latent_dim": get("irl.latent_dim"),
        "irl.num_z_samples": get("irl.num_z_samples"),
        "expert.num_trajectories": get("expert.num_trajectories"),
        "expert.confounder_value": get("expert.confounder_value"),
        "env.slip_prob": get("env.slip_prob"),
        "env.reward_type": get("env.reward_type"),
        "env.grid_size": grid_size,
        "size_N": size_N,
        "eval.heldout_region": heldout_region,
        "eval.test_z": test_z,
    }

    # Final metrics (robust extraction + standardised names)
    for base in FINAL_METRIC_BASES:
        v = _extract_final(metrics, base)
        # If base is a correlation-like metric, clamp tiny numeric overflow
        if base in ("reward_spearman", "reward_correlation", "value_correlation",
                    "policy_agreement", "value_correlation_weighted", "policy_agreement_weighted"):
            if not np.isnan(v):
                v = max(-1.0, min(1.0, v))
        info[f"final_{base}"] = v

    # ---- Weighted fallbacks (Value / Policy) ----
    if pd.isna(info.get("final_value_correlation")) and not pd.isna(info.get("final_value_correlation_weighted")):
        info["final_value_correlation"] = float(info["final_value_correlation_weighted"])
    if pd.isna(info.get("final_policy_agreement")) and not pd.isna(info.get("final_policy_agreement_weighted")):
        info["final_policy_agreement"] = float(info["final_policy_agreement_weighted"])

    # ---- Standardise invariance variance name & clamp ----
    # Keep canonical 'final_reward_variance' (already populated by the loop if present).
    # Clamp to >= 0 when finite.
    rv = info.get("final_reward_variance")
    try:
        if rv is not None and not (isinstance(rv, float) and (math.isnan(rv) or math.isinf(rv))):
            info["final_reward_variance"] = max(0.0, float(rv))
    except Exception:
        pass

    # Mirror to canonical alias expected by some plots/tables.
    # If clamped value exists, copy it; otherwise record NaN for consistency.
    fv = info.get("final_reward_variance")
    try:
        if fv is not None and not (isinstance(fv, float) and (math.isnan(fv) or math.isinf(fv))):
            info["final_reward_variance_across_z"] = float(fv)
        else:
            info["final_reward_variance_across_z"] = np.nan
    except Exception:
        info["final_reward_variance_across_z"] = np.nan

    # Per-Z sidecars (optional)
    perz = _collect_perz_values(
        run_dir,
        ("reward_spearman",
         "value_correlation",
         "policy_agreement",
         "value_correlation_weighted",
         "policy_agreement_weighted")
    )
    if perz:
        # compute worst-Z aggregates at run level (min over Z for Spearman/Value)
        sp_z: List[float] = []
        vl_z: List[float] = []
        for _, vv in perz.items():
            # Spearman (no weighted variant expected)
            sp = vv.get("reward_spearman", np.nan)
            if not pd.isna(sp):
                sp_z.append(float(sp))
            # Value correlation with weighted fallback
            vc = vv.get("value_correlation", np.nan)
            if pd.isna(vc):
                vc = vv.get("value_correlation_weighted", np.nan)
            if not pd.isna(vc):
                vl_z.append(float(vc))
        if sp_z:
            info["final_spearman_worstZ"] = float(np.min(sp_z))
        if vl_z:
            info["final_valuecorr_worstZ"] = float(np.min(vl_z))        # compute worst-Z aggregates at run level (min over Z for Spearman/Value)
        sp_z: List[float] = []
        vl_z: List[float] = []
        for _, vv in perz.items():
            # Spearman (no weighted variant expected)
            sp = vv.get("reward_spearman", np.nan)
            if not pd.isna(sp):
                sp_z.append(float(sp))
            # Value correlation with weighted fallback
            vc = vv.get("value_correlation", np.nan)
            if pd.isna(vc):
                vc = vv.get("value_correlation_weighted", np.nan)
            if not pd.isna(vc):
                vl_z.append(float(vc))
        if sp_z:
            info["final_spearman_worstZ"] = float(np.min(sp_z))
        if vl_z:
            info["final_valuecorr_worstZ"] = float(np.min(vl_z))
    return info, perz


def _collect_all_runs(roots: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      detailed_df: one row per run
      perz_df: optional per-Z rows if any were found (empty otherwise)
    """
    detailed_rows: List[Dict[str, Any]] = []
    perz_rows: List[Dict[str, Any]] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"[skip] missing root: {root}")
            continue
        if os.path.basename(root) == "scaling":
            # handled separately via perf join
            continue
        run_dirs = _discover_run_dirs(root_path)
        print(f"[discover] {root}: {len(run_dirs)} run(s)")
        for rd in run_dirs:
            try:
                row, perz = _collect_run_row(rd, root)
            except Exception as e:
                print(f"[skip] {rd}: {e}")
                continue
            detailed_rows.append(row)
            # Per-Z rows
            for z, vals in perz.items():
                prow = {
                    **{k: row.get(k) for k in (
                        "run_path", "experiment_root", "experiment_type", "method", "env_name",
                        "irl.gamma", "expert.num_trajectories", "env.slip_prob", "env.reward_type",
                        "irl.entropy_coef", "irl.grad_clip_norm",
                        "irl.kl_coeff", "irl.inv_coeff", "irl.latent_dim", "irl.num_z_samples",
                        "expert.confounder_value", "eval.test_z", "eval.heldout_region",
                        "size_N", "train.seed"
                    )},
                    "z": z,
                    "final_reward_spearman": vals.get("reward_spearman", np.nan),
                    # Value with weighted fallback at per-Z level
                    "final_value_correlation": (
                        vals.get("value_correlation", np.nan)
                        if not pd.isna(vals.get("value_correlation", np.nan))
                        else vals.get("value_correlation_weighted", np.nan)
                    ),
                    # Add per-Z policy (with weighted fallback)
                    "final_policy_agreement": (
                        vals.get("policy_agreement", np.nan)
                        if not pd.isna(vals.get("policy_agreement", np.nan))
                        else vals.get("policy_agreement_weighted", np.nan)
                    ),
                }
                perz_rows.append(prow)

    detailed_df = pd.DataFrame(detailed_rows) if detailed_rows else pd.DataFrame()
    perz_df = pd.DataFrame(perz_rows) if perz_rows else pd.DataFrame()
    return detailed_df, perz_df


def _coerce_and_normalise(detailed_df: pd.DataFrame) -> pd.DataFrame:
    if detailed_df is None or detailed_df.empty:
        return detailed_df
    # Numeric coercion
    for k in NUMERIC_SCENARIO_KEYS:
        if k in detailed_df.columns:
            detailed_df[k] = pd.to_numeric(detailed_df[k], errors="coerce")
    # Correlations already clamped in extraction; ensure numeric dtype
    for c in [col for col in detailed_df.columns if col.startswith("final_")]:
        detailed_df[c] = pd.to_numeric(detailed_df[c], errors="coerce")
    # Format slip_prob to 2 decimals (stable keys)
    if "env.slip_prob" in detailed_df.columns:
        detailed_df["env.slip_prob"] = pd.to_numeric(detailed_df["env.slip_prob"], errors="coerce").round(2)
    # Deterministic sort for diffs
    sort_keys = GROUPING_SIGNATURE + ["train.seed"]
    for key in sort_keys:
        if key not in detailed_df.columns:
            detailed_df[key] = np.nan
    detailed_df = detailed_df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    return detailed_df


def _aggregate_seed_only(detailed_df: pd.DataFrame) -> pd.DataFrame:
    if detailed_df is None or detailed_df.empty:
        return pd.DataFrame()
    # Build grouping signature, preserving all knobs (columns exist by _coerce_and_normalise)
    gb_cols = GROUPING_SIGNATURE.copy()
    # All final-* numeric metrics are aggregated
    metric_cols = [c for c in detailed_df.columns if c.startswith("final_")]
    # Compute means/std/N/ci and 95CI string
    records = []
    for sig_vals, df_g in detailed_df.groupby(gb_cols, dropna=False):
        row: Dict[str, Any] = {k: v for k, v in zip(gb_cols, sig_vals)}
        for m in metric_cols:
            xs = pd.to_numeric(df_g[m], errors="coerce").dropna().tolist()
            mean, std, N, ci, ci_str = _mean_std_ci_str(xs)
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_N"] = N
            row[f"{m}_ci95"] = ci
            row[f"{m}_95CI"] = ci_str
        # carry optional perf join if already present in detailed_df (wall_clock_s)
        if "wall_clock_s" in df_g.columns:
            row["wall_clock_s_mean"] = float(pd.to_numeric(df_g["wall_clock_s"], errors="coerce").dropna().mean())
        records.append(row)
    summary_df = pd.DataFrame.from_records(records) if records else pd.DataFrame()
    # Deterministic sort
    if not summary_df.empty:
        summary_df = summary_df.sort_values(gb_cols, kind="mergesort").reset_index(drop=True)
    return summary_df


def _maybe_join_scaling_perf(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """Optionally left-join results/scaling/perf.csv on (method, size_N, train.seed)."""
    perf_path = Path("results/scaling/perf.csv")
    if detailed_df is None or detailed_df.empty or not perf_path.exists():
        return detailed_df
    try:
        perf = pd.read_csv(perf_path)
    except Exception as e:
        print(f"[perf] failed to read {perf_path}: {e}")
        return detailed_df
    # Normalise column names we expect: method, size, seed, wall_clock_s
    # try a few variants
    meth_col = next((c for c in perf.columns if c.lower() == "method"), None)
    size_col = next((c for c in perf.columns if c.lower() in ("size", "size_n", "grid_size", "n")), None)
    seed_col = next((c for c in perf.columns if "seed" in c.lower()), None)
    time_col = next((c for c in perf.columns if "wall" in c.lower() and "s" in c.lower()), None)
    if not all([meth_col, size_col, seed_col, time_col]):
        return detailed_df
    perf_small = perf[[meth_col, size_col, seed_col, time_col]].copy()
    perf_small.columns = ["method", "size_N", "train.seed", "wall_clock_s"]
    # normalise method values
    perf_small["method"] = perf_small["method"].map(_normalise_method)
    # safe numeric coercion
    perf_small["size_N"] = pd.to_numeric(perf_small["size_N"], errors="coerce")
    perf_small["train.seed"] = pd.to_numeric(perf_small["train.seed"], errors="coerce")
    merged = detailed_df.merge(perf_small, on=["method", "size_N", "train.seed"], how="left")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment results into detailed + seed-aggregated CSVs.")
    parser.add_argument("--roots", nargs="+", default=None, help="Root directories containing run folders (can pass multiple).")
    parser.add_argument("positional_roots", nargs="*", help="(Optional) roots as positional args.")
    args = parser.parse_args()

    roots = _dedup([*(args.roots or []), *args.positional_roots]) if (args.roots or args.positional_roots) else DEFAULT_ROOTS

    # 1) Collect run-level rows (lossless) + optional per-Z
    detailed_df, perz_df = _collect_all_runs(roots)
    detailed_df = _coerce_and_normalise(detailed_df)
    detailed_df = _maybe_join_scaling_perf(detailed_df)

    # 2) Seed-only aggregation by full scenario signature
    summary_df = _aggregate_seed_only(detailed_df)

    # 3) Write outputs (deterministic paths)
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = Path("results/validation")
    validation_dir.mkdir(parents=True, exist_ok=True)

    det_csv = tables_dir / "detailed_results.csv"
    sum_csv = tables_dir / "experiment_summary.csv"
    detailed_df.to_csv(det_csv, index=False)
    summary_df.to_csv(sum_csv, index=False)

    # Back-compat mirrors for downstream scripts
    (validation_dir / "detailed_results.csv").write_text(det_csv.read_text())
    (validation_dir / "experiment_summary.csv").write_text(sum_csv.read_text())

    # 4) Optional per-Z CSV
    if perz_df is not None and not perz_df.empty:
        perz_csv = tables_dir / "perz_results.csv"
        perz_df.to_csv(perz_csv, index=False)
        # also mirror under validation for convenience
        (validation_dir / "perz_results.csv").write_text(perz_csv.read_text())

    print(f"[summarize] runs={len(detailed_df)} | scenarios={len(summary_df)} | perZ_rows={0 if perz_df is None else len(perz_df)}")
    print(f"[out] detailed  → {det_csv}")
    print(f"[out] summary   → {sum_csv}")
    if perz_df is not None and not perz_df.empty:
        print(f"[out] per-Z     → {tables_dir/'perz_results.csv'}")


if __name__ == "__main__":
    main()