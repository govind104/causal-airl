# visualisation/plot_scaling_perf.py
import argparse, os, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from visualisation.style import (
    setup_thesis_style,
    save_figure,
    method_label,
    set_method_color_cycle,
)

def mean_ci(a):
    a = np.asarray(a, float)
    n = len(a)
    if n == 0:
        return np.nan, np.nan, np.nan
    mu = float(np.mean(a))
    if n == 1:
        return mu, mu, mu
    sd = float(np.std(a, ddof=1))
    se = sd / np.sqrt(n)
    lo, hi = mu - 1.96 * se, mu + 1.96 * se
    return mu, max(lo, 0.0), hi

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=os.path.join('results', 'scaling', 'perf.csv'))
    p.add_argument('--out', required=True)
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"No scaling CSV found at {args.csv}")
        return

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"Failed to read scaling CSV: {e}")
        return
    if df.empty:
        print("No rows in scaling CSV."); return
    
    # Ensure required columns exist
    required = {"method", "size", "wall_clock_s"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Scaling CSV missing required columns: {missing}")
        return

    # Normalize columns
    df['method'] = df['method'].astype(str).str.lower()
    df['size'] = pd.to_numeric(df['size'], errors='coerce')
    df['wall_clock_s'] = pd.to_numeric(df['wall_clock_s'], errors='coerce')
    df = df.dropna(subset=['method', 'size', 'wall_clock_s'])

    setup_thesis_style()

    # Order: prefer ['ng','maxent','airl','causal_airl'] if present
    preferred = ['ng', 'maxent', 'airl', 'causal_airl']
    present = [m for m in preferred if (df['method'] == m).any()]
    others = sorted(set(df['method']) - set(present))
    methods = present + others

    set_method_color_cycle(methods)

    sizes = sorted(df['size'].dropna().unique())
    if len(sizes) == 0:
        print("No valid sizes in scaling CSV."); return

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        sub = df[df['method'] == m]
        if sub.empty:
            continue
        means, yerr_lo, yerr_hi = [], [], []
        for s in sizes:
            vals = sub.loc[sub['size'] == s, 'wall_clock_s'].values
            mu, lo, hi = mean_ci(vals)
            means.append(mu)
            # convert to asym errbars around mu
            yerr_lo.append(0.0 if np.isnan(mu) or np.isnan(lo) else (mu - lo))
            yerr_hi.append(0.0 if np.isnan(mu) or np.isnan(hi) else (hi - mu))
        x = np.array(sizes, float)
        y = np.array(means, float)
        yerr = np.vstack([yerr_lo, yerr_hi]) if np.isfinite(y).any() else None
        ax.errorbar(x, y, yerr=yerr, marker='o', capsize=3, label=method_label(m))

    ax.set_xlabel("Grid size (N for N×N)")
    ax.set_ylabel("Wall-clock (s)")
    ax.set_title("Scaling: Wall-clock vs Grid size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(args.out, exist_ok=True)
    save_figure(fig, os.path.join(args.out, "scaling_wallclock.png"))

if __name__ == "__main__":
    main()