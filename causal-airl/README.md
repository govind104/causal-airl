# Causal Deep Inverse Reinforcement Learning

This repository contains all code, experiments, and visualisations for the MSc dissertation titled:

**"Causal Inverse Reinforcement Learning for Robust Reward Recovery under Confounding"**

## 🧠 Overview

This project explores causal and deep variants of inverse reinforcement learning (IRL) in both tabular and continuous environments. We evaluate classical baselines (Ng-Russell IRL, MaxEnt IRL), adversarial models (AIRL), and a novel **Causal-AIRL** framework that infers latent confounders during reward learning. 

Environments include:
- GridWorld variants (vanilla, slippery, confounded)
- CartPole (Gymnasium continuous control)



## 📁 Code Structure

```bash
irl-project/
├── envs/           # GridWorlds, CartPole, confounding logic
├── irl/            # Ng-Russell, MaxEnt, AIRL, Causal-AIRL
├── models/         # Reward nets, policies, latent encoders
├── experiments/    # Experiment runner, sweeps, evaluation
├── visualisation/  # Plotting metrics, rewards, policies, SCMs
├── tests/          # Unit tests for IRL recovery and generalisation
├── results/        # Output figures, tables, saved models
├── make_figures.py # Reproduce all plots/tables for report
├── requirements.txt
├── environment.yaml
└── README.md
````



## ⚙️ Setup

### Option 1: pip

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option 2: conda

```bash
conda env create -f environment.yaml
conda activate causal-irl-env
```



## 🧪 Running Experiments

To run a full IRL experiment with Ng-Russell, MaxEnt, AIRL, or Causal-AIRL:

```bash
python experiments/run_experiment.py --config experiments/config_gridworld.yaml
```

Use `--seed` to specify different random initialisations.

To run multiple sweeps:

```bash
python experiments/sweeps.py
```



## 📊 Generating All Plots and Tables

After experiments are complete, generate all figures for the report:

```bash
python make_figures.py
```

Outputs will be saved to:

* `figures/` → reward heatmaps, vector fields, training curves
* `tables/` → `.csv` and `.tex` tables for Overleaf



## ✅ Reproducing Key Results

| Component             | Script / Output                                 |
| --------------------- | ----------------------------------------------- |
| Reward Heatmaps       | `plot_rewards.py` → `figures/`                  |
| Policy Visualisations | `plot_policies.py` → `figures/`                 |
| Training Curves       | `plot_training_curves.py`                       |
| Evaluation Metrics    | `plot_metrics.py`, `generate_summary_tables.py` |
| CartPole Reward       | `plot_cartpole_rewards.py`                      |
| SCM Diagrams          | `scm_diagram.py` → Fig. 4.2, 4.3                |



## 🔬 Testing

Run unit tests to verify IRL implementations:

```bash
pytest tests/
```

This validates:

* MaxEnt recovery
* AIRL reward shaping
* Causal-AIRL inference
* Generalisation to unseen confounders



## 📚 Acknowledgements

* Based on IRL methods from:

  * Ng & Russell (2000)
  * Ziebart et al. (2008)
  * Fu et al. (2018, AIRL)
* Built using PyTorch, Gymnasium, and custom GridWorld environments.



## 👤 Author

MSc Data Science, University of Edinburgh  
Author:   
Supervisor: 

Version: v1.0  
DOI:



## 📄 License

This repository is intended for academic use only.
