"""
ℹ️ About - Causal-AIRL Portfolio
==================================
Project overview, method explanation, and references.
"""

import streamlit as st
import os
import sys

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About Causal-AIRL")

st.markdown("""
## Causal Inverse Reinforcement Learning for Robust Reward Recovery

**MSc Dissertation | University of Edinburgh | 2024-25**

---

### 📖 Abstract

Inverse Reinforcement Learning (IRL) aims to recover the reward function that explains observed expert behaviour. 
Standard methods assume demonstrations arise from a single, consistent policy. However, real-world experts 
often exhibit **confounded behaviour**—actions influenced by unobserved latent factors like risk preference, 
expertise level, or contextual knowledge.

This dissertation introduces **Causal-AIRL**, an extension of Adversarial IRL that:

1. **Infers latent confounders** through a learned encoder
2. **Enforces reward invariance** via a regularization term
3. **Achieves robust generalization** across unseen confounding styles

---

### 🔬 Method: Causal-AIRL

""")

# Architecture diagram (ASCII)
st.code("""
    ┌─────────────────────────────────────────────────────────────┐
    │                    CAUSAL-AIRL ARCHITECTURE                 │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │   Expert Demos (s, a)  ──────┐                              │
    │                              │                              │
    │                              ▼                              │
    │                     ┌───────────────┐                       │
    │                     │ CausalEncoder │                       │
    │                     │   q(z|s,a)    │                       │
    │                     └───────┬───────┘                       │
    │                             │ z ~ q(z|s,a)                  │
    │                             ▼                               │
    │   ┌─────────────────────────────────────────────────────┐   │
    │   │              CausalDiscriminator                    │   │
    │   │                                                     │   │
    │   │   r_causal(s) + γ·h(s') - h(s)  =  f(s, a, s', z)   │   │
    │   │         ↑                                           │   │
    │   │    Invariant to Z                                   │   │
    │   └─────────────────────────────────────────────────────┘   │
    │                             │                               │
    │                             ▼                               │
    │                    ┌─────────────────┐                      │
    │                    │ Invariance Loss │                      │
    │                    │  Var_z[f(·,z)]  │                      │
    │                    └─────────────────┘                      │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
""", language=None)

st.markdown("""
### 📊 Key Components

| Component | Description |
|-----------|-------------|
| **CausalEncoder** | VAE-style encoder that maps (state, action) pairs to a latent z distribution |
| **CausalDiscriminator** | Discriminator + shaping potential with z-conditioned reward head |
| **Invariance Loss** | Penalizes variance of reward across sampled z values |
| **ELBO Objective** | KL regularization + reconstruction + discriminator BCE |

---

### 🧪 Experiments

| Experiment | Purpose |
|------------|---------|
| **Baselines** | Ng-Russell, MaxEnt, AIRL on standard GridWorld |
| **Hparam Sweeps** | Learning rate, entropy, KL coefficient sensitivity |
| **Scenario Sweeps** | γ, demos, slip, reward shaping effects |
| **Confounded** | Train on Z=0, test on Z=1 (and vice versa) |
| **Generalisation** | Cross-Z + held-out region evaluation |
| **Scaling** | Wall-clock vs grid size benchmarks |

---

### 📈 Key Results

""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### Cross-Z Policy Agreement
    
    When training on one confounding style and testing on another:
    
    | Method | Z=0→Z=1 | Z=1→Z=0 |
    |--------|---------|---------|
    | AIRL | 65% | 60% |
    | **Causal-AIRL** | **85%** | **82%** |
    
    *+20 percentage point improvement*
    """)

with col2:
    st.markdown("""
    #### Reward Invariance
    
    Variance of learned reward across Z samples:
    
    | Method       | Var(R&#124;Z) |
    |--------------|---------------|
    | AIRL         | 0.15          |
    | **Causal-AIRL** | **0.03**  |
    
    *5&times; lower variance = more stable reward*
    """)

st.markdown("""
---

### 🔧 Repository Structure

```
causal-airl-main/
├── envs/               # GridWorld, ConfoundedGridWorld, CartPole
├── irl/                # ng_russell, maxent_irl, airl, causal_airl
├── experiments/        # Training, evaluation, experiment runners
├── visualisation/      # Plotting scripts + Streamlit app
├── configs/            # YAML experiment configurations
├── scripts/            # Shell scripts for running experiments
└── results/            # Output directory for runs and figures
```

---

### 📚 References

1. **Fu et al. (2018)** - Learning Robust Rewards with Adversarial Inverse Reinforcement Learning
2. **Ng & Russell (2000)** - Algorithms for Inverse Reinforcement Learning  
3. **Pearl (2009)** - Causality: Models, Reasoning, and Inference
4. **Ziebart et al. (2008)** - Maximum Entropy Inverse Reinforcement Learning

---

### 👤 Author

**Govind Arun Nampoothiri**  
MSc Data Science | University of Edinburgh | Sept 2024 - Sept 2025

📂 [View on GitHub](https://github.com/govind104/causal-airl) &nbsp; · &nbsp; 💼 [LinkedIn](https://www.linkedin.com/in/govind23nampoothiri/)
""")