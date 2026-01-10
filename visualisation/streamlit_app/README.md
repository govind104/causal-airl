# Causal-AIRL Streamlit Portfolio App

Interactive demonstration of **Causal Inverse Reinforcement Learning for Robust Reward Recovery**.

## Quick Start

```bash
# From repo root
cd causal-airl-main

# Install dependencies (if not already done)
pip install -e .
pip install -r requirements.txt

# Install Streamlit-specific dependencies
pip install -r visualisation/streamlit_app/requirements.txt

# Run the app
streamlit run visualisation/streamlit_app/app.py
```

The app will open in your browser at `http://localhost:8501`.

## Pages

| Page | Description |
|------|-------------|
| **🎮 Interactive Demo** | Live GridWorld experiments with Z-toggle and method comparison |
| **🗺️ Visualisations** | Reward heatmaps, policy vector fields, training curves |
| **📊 Metrics Dashboard** | Quantitative tables, cross-Z comparison, scaling analysis |
| **ℹ️ About** | Project overview, method explanation, references |

## Requirements

- Python >= 3.10, < 3.12
- PyTorch >= 2.0
- Streamlit >= 1.28
- matplotlib, plotly, pandas, scipy

## Generating Results Data

To populate the visualisations and metrics pages, run experiments first:

```bash
# Baselines (quick)
bash scripts/run_gridworld_baselines.sh

# Confounded experiments (for cross-Z demos)
bash scripts/run_confounded_gridworld.sh

# Full sweep (optional, takes longer)
bash scripts/run_airl_scenario_sweep.sh
bash scripts/run_causal_airl_scenario_sweep.sh
```

## Project Structure

```
visualisation/streamlit_app/
├── app.py                    # Main entry point
├── pages/
│   ├── about.py
│   ├── interactive_demo.py
│   ├── metrics_dashboard.py
│   └── visualisations.py
├── utils/
│   ├── env_utils.py          # Environment creation
│   ├── model_utils.py        # Model loading with caching
│   ├── viz_utils.py          # Plotting functions
│   └── metrics_utils.py      # Metrics computation
├── assets/
│   └── style.css             # Custom styling
├── requirements.txt          # Streamlit dependencies
└── README.md                 # This file
```

## Troubleshooting

**Import errors**: Ensure you've installed the repo with `pip install -e .` from the repo root.

**No runs found**: Run experiments using the shell scripts before viewing visualisations.

**Slow policy visualisation**: The app uses vectorised inference for large grids (7×7+).

---

© 2024-25 University of Edinburgh
