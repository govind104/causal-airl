import argparse
import json
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional, Any
from matplotlib.gridspec import GridSpec
from visualisation.utils_config import find_run_dirs, load_config_with_fallback, get
from visualisation.style import setup_thesis_style, save_figure
from visualisation.scenario import label_scenario

# ---------- Defaults & small utils ----------
DEFAULT_ROOTS = [
    "results/validation",
    "results/gridworld_baselines",
    "results/airl_ablation",
    "results/causal_airl_ablation",
    "results/confounded",
    "results/generalization",
]

def _dedup(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _mkout(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _small_caption(ax, msg: str):
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="0.5"))

def _ci_mean(arr: np.ndarray) -> Tuple[float, float, float]:
    """Normal-approx 95% CI on the mean."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(arr))
    if arr.size == 1:
        return (mean, np.nan, np.nan)
    sd = float(np.std(arr, ddof=1))
    se = sd / math.sqrt(arr.size)
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    return (mean, lo, hi)

# ---------- Data loading ----------
def _load_config(run_dir: str) -> Optional[Dict]:
    try:
        return load_config_with_fallback(run_dir)
    except Exception:
        return None

def _load_training_log(run_dir: str) -> Optional[Dict[str, np.ndarray]]:
    """Robustly load training curves from training_logs.json, with fallback to metrics.json."""
    def _from_dict(d: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        # step aliases seen in our logs
        steps = (d.get("env_steps") or d.get("global_steps") or d.get("global_step")
                 or d.get("steps") or d.get("step") or d.get("iteration") or d.get("iter") or d.get("epoch"))
        if not isinstance(steps, list):
            return None
        steps = np.asarray(steps, dtype=float)
        def pick(*names):
            for n in names:
                v = d.get(n)
                if isinstance(v, list):
                    try:
                        return np.asarray(v, dtype=float)
                    except Exception:
                        continue
            return None
        # recognise common discriminator/policy aliases
        disc = pick("discriminator_loss", "disc_total_loss", "disc_bce", "D_loss")
        poli = pick("policy_loss", "pi_loss", "actor_loss")
        ret  = pick("return", "reward", "episode_return", "eval_return")
        out = {"step": steps}
        if disc is not None: out["discriminator_loss"] = disc
        if poli is not None: out["policy_loss"] = poli
        if ret  is not None: out["return"] = ret
        # if we only have steps and no series, treat as empty
        return out if len(out) > 1 else None

    # Primary: training_logs.json
    path = os.path.join(run_dir, "training_logs.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            parsed = _from_dict(data or {})
            if parsed is not None:
                return parsed
        except Exception:
            pass
    # Fallback: timelines sometimes live in metrics.json
    mpath = os.path.join(run_dir, "metrics.json")
    if os.path.exists(mpath):
        try:
            with open(mpath, "r") as f:
                md = json.load(f)
            parsed = _from_dict(md or {})
            if parsed is not None:
                return parsed
        except Exception:
            pass
    return None

def _load_true_reward(run_dir: str) -> Optional[np.ndarray]:
    # preferred sidecar
    npy = os.path.join(run_dir, "true_reward.npy")
    if os.path.exists(npy):
        try:
            return np.load(npy)
        except Exception:
            pass
    # fallback to env_data.json
    envp = os.path.join(run_dir, "env_data.json")
    if os.path.exists(envp):
        try:
            with open(envp, "r") as f:
                env = json.load(f)
            arr = np.asarray(env.get("true_reward", None), dtype=float)
            if arr is None or arr.size == 0:
                return None
            # cache for downstream reuse
            try:
                np.save(npy, arr)
            except Exception:
                pass
            return arr
        except Exception:
            return None
    return None

def _load_learned_reward(run_dir: str) -> Optional[np.ndarray]:
    for name in ("learned_reward_map.npy", "learned_reward.npy", "reward_map.npy"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            try:
                return np.load(p)
            except Exception:
                continue
    return None

def _load_grid_size_and_terminals(run_dir: str) -> Tuple[Optional[Tuple[int,int]], List[Tuple[int,int]]]:
    envp = os.path.join(run_dir, "env_data.json")
    if not os.path.exists(envp):
        return (None, [])
    try:
        with open(envp, "r") as f:
            env = json.load(f)
        H, W = env.get("grid_size", [None, None])
        terms = env.get("terminal_states") or env.get("terminals") or []
        return ((int(H), int(W)) if H is not None and W is not None else None,
                [(int(i), int(j)) for i, j in terms])
    except Exception:
        return (None, [])

def _load_tradeoff_xy(run_dir: str, x_key: str, y_key: str) -> Tuple[Optional[float], Optional[float]]:
    mp = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(mp):
        return (None, None)
    try:
        with open(mp, "r") as f:
            m = json.load(f)
        def pick(d, *names):
            for n in names:
                if n in d and d[n] is not None:
                    try:
                        return float(d[n])
                    except Exception:
                        pass
            return None
        x = pick(m, x_key, "final_wall_time_sec" if x_key!="final_wall_time_sec" else "final_env_steps")
        y = pick(m, y_key)
        return (x, y)
    except Exception:
        return (None, None)

def _compute_group_stdmap(run_dirs: List[str], grid_size: Optional[Tuple[int,int]]) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Return one representative std-map (first run that has per_z) and the group mean variance (mean of var across runs)."""
    stdmap_rep = None
    var_means = []
    for rd in run_dirs:
        perz = os.path.join(rd, "per_z")
        if not os.path.isdir(perz):
            continue
        z_maps = []
        for f in sorted(os.listdir(perz)):
            if f.startswith("reward_map_z") and f.endswith(".npy"):
                arr = np.load(os.path.join(perz, f))
                if arr.ndim == 1 and grid_size is not None:
                    H, W = grid_size
                    try:
                        arr = arr.reshape(H, W)
                    except Exception:
                        continue
                z_maps.append(arr)
        if len(z_maps) >= 2:
            stack = np.stack(z_maps, axis=0)  # (Z,H,W)
            std = stack.std(axis=0)
            var = stack.var(axis=0)
            var_means.append(float(np.mean(var)))
            if stdmap_rep is None:
                stdmap_rep = std
    if len(var_means) == 0:
        return (stdmap_rep, None)
    return (stdmap_rep, float(np.mean(var_means)))

# ---------- Plotting primitives ----------
def _apply_grid_aesthetics(ax, H: int, W: int, tick_font=8):
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.arange(W))
    ax.set_yticks(np.arange(H))
    ax.set_xticklabels([str(x) for x in range(1, W + 1)], fontsize=tick_font)
    ax.set_yticklabels([str(y) for y in range(1, H + 1)], fontsize=tick_font)
    ax.tick_params(axis='both', which='both', length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

def _plot_heatmaps(ax_true, ax_learned, true_r: np.ndarray, learned_r: np.ndarray,
                   grid_size: Tuple[int,int], terminals: List[Tuple[int,int]]):
    H, W = grid_size
    # reshape if needed
    if true_r.ndim == 1: true_r = true_r.reshape(H, W)
    if learned_r.ndim == 1: learned_r = learned_r.reshape(H, W)
    im0 = ax_true.imshow(true_r, cmap="viridis", origin="upper")
    _apply_grid_aesthetics(ax_true, H, W, tick_font=7)
    ax_true.set_title("True", fontsize=10)
    im1 = ax_learned.imshow(learned_r, cmap="viridis", origin="upper")
    _apply_grid_aesthetics(ax_learned, H, W, tick_font=7)
    ax_learned.set_title("Learned", fontsize=10)
    # overlay terminals
    for (i, j) in terminals:
        ax_true.plot(j, i, marker='*', markersize=7, markeredgewidth=1.0,
                     markeredgecolor='white', color='black')
        ax_learned.plot(j, i, marker='*', markersize=7, markeredgewidth=1.0,
                        markeredgecolor='white', color='black')
    return im0, im1

def _aggregate_curves(group_logs: List[Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    """Align by step; return mean and 95% CI per metric."""
    if not group_logs:
        return {}
    # Build union of steps
    step_sets = [set(log["step"].tolist()) for log in group_logs if "step" in log]
    if not step_sets:
        return {}
    steps = sorted(set.union(*step_sets))
    steps = np.asarray(steps)
    metrics = set()
    for log in group_logs:
        for k in ("discriminator_loss", "policy_loss", "return"):
            if k in log: metrics.add(k)
    out = {}
    for m in metrics:
        mat = np.full((len(group_logs), len(steps)), np.nan, dtype=float)
        for r, log in enumerate(group_logs):
            if m not in log or "step" not in log: continue
            s = log["step"]
            v = log[m]
            # map by step value
            idx_map = {int(s_i): i for i, s_i in enumerate(s)}
            for j, st in enumerate(steps):
                ii = idx_map.get(int(st))
                if ii is not None and ii < len(v):
                    try:
                        mat[r, j] = float(v[ii])
                    except Exception:
                        pass
        mean = np.nanmean(mat, axis=0)
        # normal-approx CI across runs
        # std over runs at each step
        std = np.nanstd(mat, axis=0, ddof=1)
        n_eff = np.sum(~np.isnan(mat), axis=0)
        se = np.divide(std, np.sqrt(np.maximum(n_eff, 1)), out=np.zeros_like(std), where=n_eff>0)
        lo = mean - 1.96 * se
        hi = mean + 1.96 * se
        out[m] = {"step": steps, "mean": mean, "lo": lo, "hi": hi, "n": n_eff}
    return out

def _plot_curves(ax, agg_curves: Dict[str, Dict[str, np.ndarray]], n_runs: int):
    if not agg_curves:
        _small_caption(ax, "slot skipped: no training logs found")
        return False
    colors = {"discriminator_loss": None, "policy_loss": None, "return": None}
    for m, series in agg_curves.items():
        s = series["step"]
        mu, lo, hi = series["mean"], series["lo"], series["hi"]
        line, = ax.plot(s, mu, label=m.replace("_", " "))
        c = line.get_color()
        ax.fill_between(s, lo, hi, alpha=0.2, linewidth=0, color=c)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title(f"Training curves (N={n_runs})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    return True

def _plot_tradeoff(ax, xs: List[float], ys: List[float], x_key: str, y_key: str):
    arrx = np.asarray(xs, dtype=float)
    arry = np.asarray(ys, dtype=float)
    arrx = arrx[~np.isnan(arrx)]
    arry = arry[~np.isnan(arry)]
    if arrx.size == 0 or arry.size == 0:
        _small_caption(ax, f"slot skipped: missing trade-off ({x_key} vs {y_key})")
        return False, None
    x_mean, x_lo, x_hi = _ci_mean(arrx)
    y_mean, y_lo, y_hi = _ci_mean(arry)
    ax.errorbar([x_mean], [y_mean],
                xerr=[[x_mean - x_lo], [x_hi - x_mean]] if not np.isnan(x_lo) else None,
                yerr=[[y_mean - y_lo], [y_hi - y_mean]] if not np.isnan(y_lo) else None,
                fmt='o', capsize=3)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.grid(True, alpha=0.25)
    text = f"{x_key}: {x_mean:.3g}\n{y_key}: {y_mean:.3g}"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top', ha='left',
            fontsize=9, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9))
    return True, {"x_mean": x_mean, "y_mean": y_mean,
                  "x_ci": [x_lo, x_hi] if not np.isnan(x_lo) else None,
                  "y_ci": [y_lo, y_hi] if not np.isnan(y_lo) else None}

def _plot_invariance_inset(ax, stdmap: Optional[np.ndarray], mean_var: Optional[float]):
    if stdmap is None and mean_var is None:
        _small_caption(ax, "slot skipped: no invariance artifacts")
        return False, None
    if stdmap is None:
        _small_caption(ax, "slot skipped: std-map unavailable")
        return False, {"mean_var": mean_var}
    H, W = stdmap.shape
    im = ax.imshow(stdmap, cmap="hot", origin="upper")
    ax.set_title(f"Invariance (var̄={ (stdmap**2).mean():.4f})", fontsize=10)
    _apply_grid_aesthetics(ax, H, W, tick_font=6)
    return True, {"mean_var": float(np.mean(stdmap**2))}

# ---------- Group & panel ----------
def _group_runs(run_dirs: List[str], scenarios_filter, methods_filter):
    groups: Dict[Tuple[str, str], List[str]] = {}
    for rd in run_dirs:
        cfg = _load_config(rd)
        if cfg is None:
            continue
        scen = label_scenario(cfg)
        meth = get(cfg, "irl.method") or get(cfg, "method") or "unknown"
        if scenarios_filter and scen not in scenarios_filter:
            continue
        if methods_filter and meth not in methods_filter:
            continue
        groups.setdefault((scen, meth), []).append(rd)
    return groups

def _panel_layout(n_slots_present: int):
    # Return (rows, cols) for GridSpec
    if n_slots_present >= 3:
        return (2, 2)
    if n_slots_present == 2:
        return (1, 2)
    return (1, 1)

_DETAILS_CACHE: Optional[pd.DataFrame] = None
def _get_detailed_results() -> Optional[pd.DataFrame]:
    """Load the first available detailed_results.csv from our default roots."""
    global _DETAILS_CACHE
    if _DETAILS_CACHE is not None:
        return _DETAILS_CACHE
    for root in DEFAULT_ROOTS:
        p = os.path.join(root, "detailed_results.csv")
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                _DETAILS_CACHE = df
                return df
            except Exception:
                continue
    return None

def _last_env_step(run_dir: str) -> Optional[float]:
    """Return the final env_steps value from logs or metrics, if present."""
    # try training logs first
    for fname in ("training_logs.json", "metrics.json"):
        p = os.path.join(run_dir, fname)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r") as f:
                d = json.load(f)
            for k in ("env_steps", "global_steps", "global_step", "steps", "step", "iteration", "iter", "epoch"):
                v = d.get(k)
                if isinstance(v, list) and len(v) > 0:
                    try:
                        return float(v[-1])
                    except Exception:
                        pass
        except Exception:
            continue
    return None

def _render_panel(group_key: Tuple[str,str], run_dirs: List[str], out_root: str,
                  x_key: str, y_key: str, aggregate: str, seed: int):
    np.random.seed(seed)
    scenario, method = group_key
    n_runs = len(run_dirs)

    # Curves
    logs = [_load_training_log(rd) for rd in run_dirs]
    logs = [l for l in logs if l is not None]
    agg_curves = _aggregate_curves(logs)
    has_curves = len(agg_curves) > 0

    # Heatmaps
    # Use the first run in group that has both arrays & env shape
    true_arr, learned_arr, grid_size, terminals = None, None, None, []
    for rd in run_dirs:
        tr = _load_true_reward(rd)
        lr = _load_learned_reward(rd)
        gs, terms = _load_grid_size_and_terminals(rd)
        if tr is not None and lr is not None and gs is not None:
            true_arr, learned_arr, grid_size, terminals = tr, lr, gs, terms
            break
    has_heatmaps = (true_arr is not None and learned_arr is not None and grid_size is not None)

    # Trade-off
    # X = last env_steps per run; Y = from detailed_results.csv filtered by (scenario, method)
    xs, ys = [], []
    for rd in run_dirs:
        xe = _last_env_step(rd)
        if xe is not None:
            xs.append(xe)
    df = _get_detailed_results()
    if df is not None and y_key in df.columns:
        try:
            # filter by scenario/method (these columns are present in our tables)
            mask = (df.get("scenario") == scenario) & (df.get("method") == method)
            col = pd.to_numeric(df.loc[mask, y_key], errors="coerce").dropna()
            ys = col.tolist()
        except Exception:
            ys = []
    has_tradeoff = (len(xs) > 0 and len(ys) > 0)

    # Invariance (group)
    stdmap, mean_var = _compute_group_stdmap(run_dirs, grid_size)
    has_invariance = (stdmap is not None or mean_var is not None)

    available = [has_curves, has_heatmaps, has_tradeoff, has_invariance]
    n_present = sum(1 for v in available if v)
    rows, cols = _panel_layout(n_present if n_present>0 else 1)

    setup_thesis_style()
    fig = plt.figure(figsize=(10.5, 7.5))
    gs = GridSpec(rows, cols, figure=fig)

    # Give the suptitle and the top row some breathing room
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.10, top=0.90)

    axes = []
    # Map desired slot order A,B,C, D into available grid
    # We fill in reading order
    for r in range(rows):
        for c in range(cols):
            axes.append(fig.add_subplot(gs[r, c]))
    # Helper to pick next axis & render or caption
    slot_flags = {"curves": False, "heatmaps": False, "tradeoff": False, "invariance": False}
    ax_iter = iter(axes)
    sidecar = {"scenario": scenario, "method": method, "n_runs": n_runs,
               "tradeoff": None, "invariance": {"mean_var": None}}

    # (A) Curves
    ax = next(ax_iter)
    if has_curves:
        slot_flags["curves"] = _plot_curves(ax, agg_curves, n_runs)
    else:
        _small_caption(ax, "slot skipped: no training logs found")

    # (B) Heatmaps
    # Choose a container axis:
    #  - if we only have one slot total *and* heatmaps are the only content,
    #    reuse the first axis (clear any caption drawn on it);
    #  - otherwise, use the next grid cell when available.
    container_ax = None
    total_cells = rows * cols
    if total_cells == 1:
        if has_heatmaps and (not has_curves and not has_tradeoff and not has_invariance):
            container_ax = ax
            ax.cla()  # clear the "no training logs" caption drawn earlier
    else:
        container_ax = next(ax_iter)

    if container_ax is None:
        # No place to draw heatmaps in the current layout
        if not has_heatmaps:
            _small_caption(ax, "slot skipped: no heatmaps")
    else:
        if has_heatmaps:
            # create two mini-axes inside the chosen container axis
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            left = inset_axes(container_ax, width="48%", height="90%",
                              loc="center left", borderpad=1.0)
            right = inset_axes(container_ax, width="48%", height="90%",
                               loc="center right", borderpad=1.0)
            _plot_heatmaps(left, right, true_arr, learned_arr, grid_size, terminals)
            container_ax.set_title("Rewards", fontsize=11, pad=4)
            container_ax.axis("off")
            slot_flags["heatmaps"] = True
        else:
            _small_caption(container_ax, "slot skipped: no heatmaps")

    # (C) Trade-off
    if rows*cols >= 3:
        axC = next(ax_iter)
        ok, trade = _plot_tradeoff(axC, xs, ys, x_key, y_key)
        slot_flags["tradeoff"] = ok
        if trade:
            sidecar["tradeoff"] = trade
    elif not has_tradeoff:
        pass
    else:
        # not enough room; annotate on curves panel
        pass

    # (D) Invariance inset
    if rows*cols >= 4:
        axD = next(ax_iter)
        ok, inv = _plot_invariance_inset(axD, stdmap, mean_var)
        slot_flags["invariance"] = ok
        if inv and "mean_var" in inv:
            sidecar["invariance"]["mean_var"] = inv["mean_var"]
    elif has_invariance and sidecar["invariance"]["mean_var"] is None and mean_var is not None:
        sidecar["invariance"]["mean_var"] = mean_var

    # Title + subtitle
    shown = ",".join([k for k,v in slot_flags.items() if v])
    skipped = ",".join([k for k,v in slot_flags.items() if not v])
    fig.suptitle(f"{scenario} — {method} (N={n_runs})", fontsize=13)
    fig.text(0.5, 0.965, f"slots shown: [{shown}]  |  skipped: [{skipped}]",
             ha="center", va="top", fontsize=9)

    # Save
    out_png = os.path.join(out_root, f"{scenario}__{method}__panel.png")
    _mkout(out_png)
    save_figure(fig, out_png, tight=False)
    plt.close(fig)
    # Sidecar JSON
    sidecar_path = os.path.join(out_root, f"{scenario}__{method}__panel.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    # Log
    print(f"[panel] {scenario}, {method}, N={n_runs}; slots: curves={slot_flags['curves']}, "
          f"heatmaps={slot_flags['heatmaps']}, tradeoff={slot_flags['tradeoff']}, invariance={slot_flags['invariance']} -> {out_png}")

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Summary (scenario × method) panels: curves, rewards, tradeoff, invariance")
    parser.add_argument("--roots", nargs="+", default=[], help="Root directories with runs")
    parser.add_argument("--out", required=True, help="Output directory, e.g., results/figures/panels")
    parser.add_argument("--metric", default="final_reward_correlation",
                        help="Y metric for trade-off (default: final_reward_correlation)")
    parser.add_argument("--x", dest="x_key", default="final_env_steps",
                        help="X axis for trade-off (default: final_env_steps)")
    parser.add_argument("--aggregate", choices=["mean","best"], default="mean",
                        help="Aggregation mode for curves/tradeoff (currently mean supported)")
    parser.add_argument("--scenarios", nargs="*", default=None, help="Optional scenario filter(s)")
    parser.add_argument("--methods", nargs="*", default=None, help="Optional method filter(s)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for CI/bootstrap")
    args = parser.parse_args()

    eff_roots = _dedup(list(args.roots) + DEFAULT_ROOTS)
    run_dirs = find_run_dirs(eff_roots)
    print(f"[inputs] roots={eff_roots} -> discovered {len(run_dirs)} runs")
    if len(run_dirs) == 0:
        os.makedirs(args.out, exist_ok=True)
        # Save a placeholder figure so pipelines remain deterministic
        fig = plt.figure(figsize=(6, 3))
        _small_caption(plt.gca(), "No runs discovered under roots")
        save_figure(fig, os.path.join(args.out, "no_runs_found.png"), tight=True)
        return

    groups = _group_runs(run_dirs, set(args.scenarios) if args.scenarios else None,
                         set(args.methods) if args.methods else None)
    if not groups:
        fig = plt.figure(figsize=(6, 3))
        _small_caption(plt.gca(), "No groups after filters (scenario/method)")
        save_figure(fig, os.path.join(args.out, "no_groups_after_filters.png"), tight=True)
        print("[warn] no groups after filters")
        return

    os.makedirs(args.out, exist_ok=True)
    for key, rds in sorted(groups.items()):
        if len(rds) == 0:
            continue
        _render_panel(key, rds, args.out, args.x_key, args.metric, args.aggregate, args.seed)

if __name__ == "__main__":
    main()
