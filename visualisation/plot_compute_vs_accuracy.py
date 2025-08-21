import argparse
import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from visualisation.utils_config import find_run_dirs, flatten_config, get
from visualisation.style import setup_thesis_style, save_figure, method_label, set_method_color_cycle

CANDIDATE_KEYS = {
    "reward_spearman": ["final_reward_spearman", "reward_spearman"],
    "reward_correlation": ["final_reward_correlation", "reward_correlation", "reward_corr"],
    "policy_agreement": ["final_policy_agreement", "policy_agreement",
                         "final_policy_agreement_weighted", "policy_agreement_weighted"],
    "value_correlation": ["final_value_correlation", "value_correlation",
                          "final_value_correlation_weighted", "value_correlation_weighted"],
}

GRID_RE = re.compile(r"\[(\d+),\s*\1\]")

def _try_get_metric(metrics: dict, metric: str):
    keys = CANDIDATE_KEYS.get(metric, [metric])
    for k in keys:
        if k in metrics and metrics[k] is not None:
            try:
                return float(metrics[k])
            except Exception:
                pass
    return None

def _extract_size(cfg_flat: dict):
    # Prefer numeric; fallback to string like "[7,7]"
    gs = get(cfg_flat, "env.grid_size", None)
    if isinstance(gs, (list, tuple)) and len(gs) == 2 and gs[0] == gs[1]:
        return int(gs[0])
    if isinstance(gs, str):
        m = GRID_RE.match(gs.strip())
        if m:
            return int(m.group(1))
    # Not found
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--perf", default=os.path.join("results", "scaling", "perf.csv"))
    p.add_argument("--roots", default=os.path.join("results", "scaling"))
    p.add_argument("--metric", default="reward_spearman",
                   choices=list(CANDIDATE_KEYS.keys()))
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if not os.path.exists(args.perf):
        print(f"[compute_vs_accuracy] Missing perf CSV: {args.perf}")
        return

    try:
        perf = pd.read_csv(args.perf)
    except Exception as e:
        print(f"[compute_vs_accuracy] Failed to read perf CSV: {e}")
        return
    if perf.empty:
        print("[compute_vs_accuracy] Empty perf CSV.")
        return

    # Ensure required columns exist before coercions
    required = {"method", "size", "seed", "wall_clock_s"}
    missing = [c for c in required if c not in perf.columns]
    if missing:
        print(f"[compute_vs_accuracy] perf CSV missing required columns: {missing}")
        return

    perf["method"] = perf["method"].astype(str).str.lower()
    perf["size"] = pd.to_numeric(perf["size"], errors="coerce")
    perf["seed"] = pd.to_numeric(perf["seed"], errors="coerce")
    perf["wall_clock_s"] = pd.to_numeric(perf["wall_clock_s"], errors="coerce")
    perf = perf.dropna(subset=["method", "size", "seed", "wall_clock_s"])

    # Discover scaling runs; we expect them under results/scaling/<method>/... with metrics.json
    discovered = find_run_dirs([args.roots])
    # Build an index of candidate run dirs per method for quick filtering
    by_method = {}
    for rd in discovered:
        # Try config_flat first (faster); fallback to config
        cfg_path = os.path.join(rd, "config_flat.json")
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(rd, "config.json")
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            cfg_flat = cfg if "env.grid_size" in cfg else flatten_config(cfg)
        except Exception:
            continue
        method = get(cfg_flat, "irl.method", "unknown")
        if method is None:
            continue
        method = str(method).lower()
        by_method.setdefault(method, []).append((rd, cfg_flat))

    # For each (method,size,seed) in perf, find the best match run and extract the metric
    rows = []
    for (m, s, seed, wc) in perf[["method", "size", "seed", "wall_clock_s"]].itertuples(index=False, name=None):
        cands = by_method.get(m, [])
        best = None
        best_mtime = -1
        for rd, cfg_flat in cands:
            # seed match
            cfg_seed = get(cfg_flat, "train.seed", None)
            if cfg_seed is None:
                continue
            try:
                cfg_seed = int(cfg_seed)
            except Exception:
                continue
            if cfg_seed != int(seed):
                continue
            # size match
            sz = _extract_size(cfg_flat)
            if sz is None or int(sz) != int(s):
                continue
            mp = os.path.join(rd, "metrics.json")
            if not os.path.exists(mp):
                continue
            mt = os.path.getmtime(mp)
            if mt > best_mtime:
                best_mtime = mt
                best = (rd, mp)
        metric_val = None
        if best is not None:
            try:
                with open(best[1]) as f:
                    metrics = json.load(f)
                metric_val = _try_get_metric(metrics, args.metric)
            except Exception:
                metric_val = None
        if metric_val is not None:
            rows.append(dict(method=m, size=int(s), seed=int(seed), wall_clock_s=float(wc), metric=float(metric_val)))

    if not rows:
        print("[compute_vs_accuracy] No matching runs with metrics found; nothing to plot.")
        return

    df = pd.DataFrame(rows)

    setup_thesis_style()

    # Stable method ordering & colors
    preferred = ["ng", "maxent", "airl", "causal_airl"]
    methods = [m for m in preferred if (df["method"] == m).any()] + \
              [m for m in sorted(df["method"].unique()) if m not in preferred]
    set_method_color_cycle(methods)

    # Marker set for sizes (repeat if necessary)
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    sizes = sorted(df["size"].unique())
    size_to_marker = {sz: markers[i % len(markers)] for i, sz in enumerate(sizes)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        for sz in sizes:
            pts = sub[sub["size"] == sz]
            if pts.empty:
                continue
            ax.scatter(pts["wall_clock_s"], pts["metric"],
                       marker=size_to_marker[sz], alpha=0.8, label=f"{method_label(m)} (N={sz})")
        # mean line per method across size (aggregate by size)
        means = sub.groupby("size", as_index=False)[["wall_clock_s", "metric"]].mean(numeric_only=True)
        means = means.sort_values("wall_clock_s")
        if len(means) >= 2:
            ax.plot(means["wall_clock_s"], means["metric"], linewidth=1.0, alpha=0.6)

    # De-duplicate legend (since we add one label per size)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    new_h, new_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            new_h.append(h); new_l.append(l)
    ax.legend(new_h, new_l, ncol=2)

    ax.set_xlabel("Wall-clock (s)")
    ax.set_ylabel(args.metric.replace("_", " ").title())
    ax.set_title(f"Compute vs Accuracy ({args.metric})")
    ax.grid(True, alpha=0.3)

    os.makedirs(args.out, exist_ok=True)
    save_figure(fig, os.path.join(args.out, "compute_vs_accuracy.png"))

if __name__ == "__main__":
    main()