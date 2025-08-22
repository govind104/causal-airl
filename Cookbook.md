
# Figures Cookbook

Minimal, copy-pasteable recipes. Each item: **what it shows → how to generate → what to report → one-line expectation → caption template**.

> Conventions used below
> - `<RUN_ROOTS...>` = one or more run directories (space-separated) that contain `config*.json`, `metrics.json`, etc.
> - `<RUN_DIR>` = a single run directory.
> - `<PARENT_DIR>` = a folder containing many `<RUN_DIR>` children.
> - All commands run from the repo root. Outputs are written under `results/figures` (create if missing).

## Quick start (aggregate)
Make everything:
```bash
python -m visualisation.make_all --out results/figures
```

Run a single module (fast for spot checks):

```bash
python -m visualisation.make_all --only plot_rewards --out results/figures
# or skip specific ones
python -m visualisation.make_all --skip plot_training_curves --out results/figures
```

---

## 1) Reward maps (true vs learned vs diff) — `visualisation/plot_rewards.py`

**Shows:** spatial structure of recovered reward and deviations from ground truth.
**Make-all:** `--only plot_rewards`
**Direct:**

```bash
python -m visualisation.plot_rewards --roots <RUN_ROOTS...> --out results/figures/reward
```

**Report:** `final_reward_spearman` (train/test if available), `final_reward_correlation`, `expert.num_trajectories`, `gamma`.
**Expect:** CAIRL smoother maps; AIRL shows off-manifold artefacts.
**Caption template:** *Setup:* GridWorld, demos=<N>, γ=<γ>. *Metric:* Spearman (train/test). *Expected:* CAIRL > AIRL on test. *Observed:* <…>. *Takeaway:* <…>.

---

## 2) Held-out region overlay — `visualisation/plot_heldout_overlay.py`

**Shows:** evaluation mask for unseen states over the grid.
**Make-all:** `--only plot_heldout_overlay`
**Direct:**

```bash
python -m visualisation.plot_heldout_overlay --roots <RUN_ROOTS...> --out results/figures/heldout
```

**Report:** `reward_spearman_train`, `reward_spearman_test`, Δ(test−train).
**Expect:** CAIRL smaller generalisation gap.
**Caption:** *Setup:* held-out mask overlay. *Metric:* train/test Spearman, gap. *Expected:* smaller gap for CAIRL. *Observed:* <…>. *Takeaway:* <…>.

---

## 3) Cross-Z generalisation bars — `visualisation/plot_crossz_bars.py`

**Shows:** performance when training on z=a and testing on z=b (and vice-versa).
**Make-all:** `--only plot_crossz_bars`
**Direct:**

```bash
python -m visualisation.plot_crossz_bars --roots <RUN_ROOTS...> --out results/figures/crossz
```

**Report:** cross-Z agreement numbers (means ± CI if present), N.
**Expect:** CAIRL ≈ invariant across z; AIRL asymmetric drop.
**Caption:** *Setup:* confounded demos across z. *Metric:* cross-Z agreement (±CI). *Expected:* flat bars for CAIRL. *Observed:* <…>. *Takeaway:* <…>.

---

## 4) Reward invariance heatmap — `visualisation/plot_causal_invariance.py`

**Shows:** variance/STD of reward across Z (per run or across runs).
**Make-all:** `--only plot_causal_invariance`
**Direct (examples):**

```bash
# per run
python -m visualisation.plot_causal_invariance --mode per_run --run_dir <RUN_DIR> --out results/figures/invariance
# across runs (group by confounder value)
python -m visualisation.plot_causal_invariance --mode across_runs --roots <RUN_ROOTS...> --groupby env.confounder_value --out results/figures/invariance
```

**Report:** variance/STD scalar, any summary printed.
**Expect:** CAIRL low variance; AIRL higher in off-manifold regions.
**Caption:** *Setup:* variance over Z. *Metric:* var/STD across per-Z maps. *Expected:* CAIRL ≪ AIRL. *Observed:* <…>. *Takeaway:* <…>.

---

## 5) Invariance violation markers — `visualisation/plot_invariance_violations.py`

**Shows:** locations where invariance breaks (from analysis JSON or per-Z STD).
**Make-all:** `--only plot_invariance_violations`
**Direct:**

```bash
python -m visualisation.plot_invariance_violations --roots <RUN_ROOTS...> --out results/figures/inv_violations
```

**Report:** number of violating sites K; max/mean violation magnitude.
**Expect:** Fewer/weaker violations for CAIRL.
**Caption:** *Setup:* violations over reward map. *Metric:* K, magnitude. *Expected:* CAIRL fewer/lower. *Observed:* <…>. *Takeaway:* <…>.

---

## 6) Training curves — `visualisation/plot_training_curves.py`

**Shows:** optimisation stability (discriminator/π losses; invariance loss when present).
**Make-all:** `--only plot_training_curves`
**Direct:**

```bash
python -m visualisation.plot_training_curves --roots <RUN_ROOTS...> --metrics discriminator_loss policy_loss epoch_inv_loss --out results/figures/training
```

**Report:** final/median `policy_agreement`, `discriminator_loss`, `epoch_inv_loss`, `final_wall_time_sec`.
**Expect:** CAIRL steadier; AIRL oscillations at higher clip.
**Caption:** *Setup:* <config>. *Metric:* losses/agreement vs steps. *Expected:* steadier CAIRL. *Observed:* <…>. *Takeaway:* <…>.

---

## 7) Ablation summary heatmaps — `visualisation/plot_ablation_summaries.py`

**Shows:** sensitivity to γ, demos, slip, reward type by method/scenario.
**Make-all:** (called automatically in ablation presets)
**Direct (example):**

```bash
python -m visualisation.plot_ablation_summaries \
  --roots <PARENT_DIR> \
  --x expert.num_trajectories \
  --metrics final_reward_spearman final_policy_agreement \
  --groupby env.slip_prob irl.method \
  --out results/figures/ablations/x=demos
```

**Report:** chosen y-metrics (test variants if available), group labels, best settings.
**Expect:** ↑ with demos; CAIRL less sensitive to slip/shaping.
**Caption:** *Setup:* factorial sweep. *Metric:* test fidelity/agree. *Expected:* monotone in demos; CAIRL robust to slip. *Observed:* <…>. *Takeaway:* <…>.

---

## 8) Checkpoint evolution — `visualisation/plot_checkpoint_evolution.py`

**Shows:** reward map progression across checkpoints + metric timeline.
**Make-all:** `--only plot_checkpoint_evolution`
**Direct:**

```bash
python -m visualisation.plot_checkpoint_evolution --roots <RUN_ROOTS...> --out results/figures/checkpoints
```

**Report:** iterations sampled; correlation at those iterations.
**Expect:** CAIRL stabilises earlier.
**Caption:** *Setup:* K checkpoints. *Metric:* correlation timeline. *Expected:* earlier stabilisation for CAIRL. *Observed:* <…>. *Takeaway:* <…>.

---

## 9) Compute vs accuracy (per-scenario trade-offs) — `visualisation/plot_tradeoffs.py`

**Shows:** wall-time/env-steps vs accuracy, faceted by scenario.
**Make-all:** `--only plot_tradeoffs`
**Direct:**

```bash
python -m visualisation.plot_tradeoffs --roots <RUN_ROOTS...> --out results/figures/tradeoffs
```

**Report:** x=`final_wall_time_sec` or `final_env_steps`; y=`final_reward_spearman`/`final_policy_agreement`; scenario facet.
**Expect:** CAIRL sits on/above AIRL frontier in confounded regimes.
**Caption:** *Setup:* same hardware. *Metric:* y vs x. *Expected:* CAIRL efficient frontier. *Observed:* <…>. *Takeaway:* <…>.

---

## 10) Compute vs accuracy (scaling CSV) — `visualisation/plot_compute_vs_accuracy.py`

**Shows:** runtime–accuracy scatter from scaling runs/perf.csv.
**Make-all:** `--only plot_compute_vs_accuracy`
**Direct:**

```bash
python -m visualisation.plot_compute_vs_accuracy \
  --roots results/scaling \
  --perf results/scaling/perf.csv \
  --metric reward_spearman \
  --out results/figures/compute_vs_accuracy
```

**Report:** chosen perf metric vs wall-time; annotate method.
**Expect:** CAIRL competitive at equal/less compute.
**Caption:** *Setup:* scaling runs. *Metric:* chosen perf vs time. *Expected:* CAIRL on frontier. *Observed:* <…>. *Takeaway:* <…>.

---

## 11) Reward statistics panel — `visualisation/plot_reward_stats.py`

**Shows:** sparsity, Gini(|R|), entropy(|R|), std, skewness across methods/scenarios.
**Make-all:** `--only plot_reward_stats`
**Direct:**

```bash
python -m visualisation.plot_reward_stats --roots <RUN_ROOTS...> --out results/figures/reward_stats
```

**Report:** the five stats computed from `learned_reward*_map.npy`.
**Expect:** CAIRL lower sparsity/Gini (less brittle) at similar entropy.
**Caption:** *Setup:* per-scenario method comparison. *Metric:* five reward stats. *Expected:* CAIRL less brittle. *Observed:* <…>. *Takeaway:* <…>.

---

## 12) AIRL vs CAIRL differences — `visualisation/plot_reward_differences.py`

**Shows:** |R\_AIRL − R\_CAIRL| (or signed) with terminals inset.
**Make-all:** `--only plot_reward_differences`
**Direct:**

```bash
python -m visualisation.plot_reward_differences --roots <RUN_ROOTS...> --out results/figures/diff
```

**Report:** scenario, terminals; highlight largest deltas.
**Expect:** largest diffs off demo manifold; CAIRL suppresses spurious peaks.
**Caption:** *Setup:* paired runs per scenario. *Metric:* |ΔR| heatmap. *Expected:* AIRL deviates off-manifold. *Observed:* <…>. *Takeaway:* <…>.

---

## 13) Policy vector fields — `visualisation/plot_policy_fields.py`

**Shows:** greedy action field from `policy.pt`; optional trajectory overlays.
**Make-all:** `--only plot_policy_fields`
**Direct:**

```bash
python -m visualisation.plot_policy_fields --roots <RUN_ROOTS...> --overlay_trajectories --out results/figures/policy
```

**Report:** `final_policy_agreement` for caption; mention overlays.
**Expect:** CAIRL smoother, goal-directed fields in low-coverage zones.
**Caption:** *Setup:* learned π; overlay trajectories. *Metric:* agreement. *Expected:* smoother CAIRL. *Observed:* <…>. *Takeaway:* <…>.

---

## 14) Trajectory diversity — `visualisation/plot_trajectory_diversity.py`

**Shows:** `trajectory_entropy`, `trajectory_overlap`; path length & coverage when available.
**Make-all:** `--only plot_trajectory_diversity`
**Direct:**

```bash
python -m visualisation.plot_trajectory_diversity --roots <RUN_ROOTS...> --out results/figures/trajectories
```

**Report:** trajectory entropy/overlap (last values), episodes, coverage.
**Expect:** CAIRL higher entropy at similar overlap (more varied yet consistent).
**Caption:** *Setup:* eval trajectories. *Metric:* entropy/overlap. *Expected:* CAIRL higher entropy. *Observed:* <…>. *Takeaway:* <…>.

---

## 15) Scaling performance — `visualisation/plot_scaling_perf.py`

**Shows:** wall-clock vs problem size from `results/scaling/perf.csv`.
**Direct:**

```bash
python -m visualisation.plot_scaling_perf --csv results/scaling/perf.csv --out results/figures/scaling
```

**Report:** mean ± CI per method/size bin.
**Expect:** CAIRL scales near AIRL with better quality/compute balance.
**Caption:** *Setup:* scaling benchmark. *Metric:* wall-clock vs size. *Expected:* CAIRL competitive scaling. *Observed:* <…>. *Takeaway:* <…>.

---

## 16) Summary tables (CSV/Markdown) — `visualisation/generate_summary_tables.py`

**Shows:** bootstrap means + CIs for chosen metrics across groupings.
**Make-all:** `--only generate_summary_tables`
**Direct:**

```bash
python -m visualisation.generate_summary_tables \
  --csv results/validation/experiment_summary.csv \
  --out results/figures/tables/summary.csv \
  --md results/figures/tables/summary.md
```

**Report:** metrics selected (defaults cover reward/policy/value correlations and cross-Z).
**Expect:** CAIRL best on confounded/generalisation rows.
**Caption (if converted to figure):** use as table notes under the Results overview.

---

## Metrics key quick-ref (use consistent names in captions)

* `reward_spearman_train/test` – Spearman correlation with ground-truth reward (statewise).
* `policy_agreement_train/test` – action agreement with expert policy.
* `return_mean/return_std` – episodic returns under learned policy.
* `invariance_gap` or var/STD across Z – causal robustness proxy.
* `final_wall_time_sec`, `final_env_steps` – compute budget proxies.

## Command patterns / tips

* Aggregator first for convenience:

```bash
  python -m visualisation.make_all --out results/figures
```

* Target a subset with `--only` or exclude with `--skip`.
* For sweeps, pass the **parent directory** to `--roots` so the script discovers many runs.
* Most scripts infer CI/SE when multiple seeds exist; otherwise, rely on table generation for CIs.
