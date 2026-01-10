"""
Causal-AIRL Streamlit Portfolio App
=====================================
Interactive demonstration of Causal Inverse Reinforcement Learning
for Robust Reward Recovery.

Main entry point with multi-page navigation.
Enhanced landing page with key visualisations.
"""

import streamlit as st
import numpy as np
import os
import sys

# Add repo root to path for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Causal-AIRL Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Safe imports
VIZ_OK = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    VIZ_OK = True
except ImportError:
    pass

# Custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Sidebar branding
with st.sidebar:
    st.title("🧠 Causal-AIRL")
    st.caption("23pp Improvement in Cross-Z Generalization via Causal Reward Learning")
    st.divider()
    st.markdown("**Author:** \n\n Govind Arun Nampoothiri")
    st.markdown("MSc Data Science | University of Edinburgh | 2024-25")
    st.markdown("📂 [GitHub Repository](https://github.com/govind104/causal-airl)")
    st.markdown("💼 [LinkedIn](https://www.linkedin.com/in/govind23nampoothiri/)")

# ============================================================================
# HERO SECTION
# ============================================================================

st.title("🧠 Causal Inverse Reinforcement Learning")
st.subheader("For Robust Reward Recovery Under Confounded Expert Demonstrations")

# TL;DR hook
st.info("**TL;DR:** I built an IRL method that infers hidden expert preferences (Z) and learns rewards that generalize **23% better** across different expert styles.")

st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 20px; border-radius: 10px; color: white; margin: 20px 0;">
<h4 style="color: white; margin-top: 0;">📚 MSc Dissertation | University of Edinburgh | 2024-25</h4>
<p style="margin-bottom: 0;">Standard IRL fails when experts exhibit <b>style variations</b> from unobserved factors.<br>
Causal-AIRL recovers <b>Z-invariant rewards</b> that generalize across confounding styles.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# KEY RESULTS - VISUAL (THE HOOK)
# ============================================================================

st.header("📊 Key Results at a Glance")

col1, col2 = st.columns([1.2, 1])

with col1:
    # Cross-Z comparison chart - THE key result
    if VIZ_OK:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        labels = ['Train Z=0\nTest Z=1', 'Train Z=1\nTest Z=0']
        airl_vals = [0.65, 0.60]
        cairl_vals = [0.88, 0.85]
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, airl_vals, width, label='AIRL', color='#D55E00', edgecolor='black')
        bars2 = ax.bar(x + width/2, cairl_vals, width, label='Causal-AIRL', color='#0072B2', edgecolor='black')
        
        ax.set_ylabel('Policy Agreement', fontsize=11)
        ax.set_title('Cross-Z Generalization (The Key Test)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.4, linewidth=1)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Annotations
        for bars, color in [(bars1, '#8B0000'), (bars2, '#00008B')]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 4), textcoords="offset points",
                           ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Install matplotlib to see visualisations")

with col2:
    st.markdown("""
    ### 🎯 The Problem
    Standard IRL assumes consistent expert behaviour. 
    Real experts have **unobserved preferences** (Z) that affect their actions.
    
    ### 💡 The Solution
    **Causal-AIRL** infers Z from (state, action) pairs and learns a reward that is
    **invariant to Z** — capturing *what* the expert wants, not *how* they happened to do it.
    
    ### ✅ The Result
    **+23 percentage point** improvement in cross-Z policy agreement!
    """)


# ============================================================================
# KEY METRICS CARDS
# ============================================================================

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Cross-Z Improvement", "+23 p.p.", delta="AIRL 62.5% → CAIRL 86.5%", delta_color="inverse")
with col2:
    st.metric("Reward Var(Z)", "5× lower", delta="0.15 → 0.03", delta_color="inverse")
with col3:
    st.metric("Clean Data", "Same perf", help="No regression on data without confounding")
with col4:
    st.metric("Methods Tested", "4", help="Ng-Russell, MaxEnt, AIRL, Causal-AIRL")


# ============================================================================
# METHOD COMPARISON TABLE
# ============================================================================

st.divider()
st.header("📈 Full Comparison (Confounded GridWorld)")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("""
    | Metric | AIRL | Causal-AIRL | Δ |
    |--------|------|-------------|---|
    | **Cross-Z Agreement** | 62.5% | 86.5% | **+24 p.p.** |
    | **Reward Spearman** | 0.72 | 0.91 | +0.19 |
    | **Policy Agreement** | 65% | 87% | +22 p.p. |
    | **Reward Variance** | 0.15 | 0.03 | 5× lower |
    
    *Test setting: 5×5 ConfoundedGridWorld, sparse reward, 20 demos*
    """)

with col2:
    # Method comparison bar chart
    if VIZ_OK:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        
        metrics = ['Reward\nSpearman', 'Policy\nAgreement', 'Cross-Z\nAgreement']
        airl = [0.72, 0.65, 0.63]
        cairl = [0.91, 0.87, 0.86]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, airl, width, label='AIRL', color='#D55E00', edgecolor='black')
        ax.bar(x + width/2, cairl, width, label='Causal-AIRL', color='#0072B2', edgecolor='black')
        
        ax.set_ylabel('Score')
        ax.set_title('Method Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ============================================================================
# HOW IT WORKS (QUICK)
# ============================================================================

st.divider()
st.header("🔧 How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1️⃣ Encode Z
    ```
    z ~ q(z | s, a)
    ```
    CausalEncoder infers latent confounder from state-action pairs.
    """)

with col2:
    st.markdown("""
    ### 2️⃣ Learn Reward
    ```
    f(s,a,s',z) = r(s) + γh(s') - h(s)
    ```
    Discriminator with potential-based shaping.
    """)

with col3:
    st.markdown("""
    ### 3️⃣ Enforce Invariance
    ```
    L_inv = Var_z[ f(·|z) ]
    ```
    Penalize reward variance across Z samples.
    """)


# ============================================================================
# NAVIGATION CTA
# ============================================================================

st.divider()

st.markdown("""
<div style="background: #e8f4f8; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #b8d4e3;">
<h3 style="color: #1a1a2e; margin-top: 0;">👈 Explore the full portfolio using the sidebar</h3>
<p style="color: #333;"><b>🎮 Interactive Demo</b> — Simulate expert trajectories and confounding<br>
<b>🗺️ Visualisations</b> — Reward heatmaps and policy fields<br>
<b>📊 Metrics Dashboard</b> — Detailed quantitative comparisons<br>
<b>ℹ️ About</b> — Architecture details and references</p>
</div>
""", unsafe_allow_html=True)