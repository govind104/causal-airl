"""
📊 Metrics Dashboard - Causal-AIRL Portfolio
==============================================
Quantitative comparison: Cross-Z bars, summary table, key insights.
Uses dissertation-validated demo data by default for accurate representation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import sys

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

st.set_page_config(page_title="Metrics Dashboard", page_icon="📊", layout="wide")

st.title("📊 Metrics Dashboard")
st.markdown("Quantitative comparison of AIRL vs Causal-AIRL from the dissertation experiments.")

# Safe imports
VIZ_OK = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    VIZ_OK = True
except ImportError:
    pass


# ============================================================================
# DISSERTATION-VALIDATED DEMO DATA
# ============================================================================

def get_demo_metrics(reward_type="all"):
    """Return dissertation-validated metrics."""
    data = {
        'Method': ['AIRL', 'AIRL', 'AIRL', 'AIRL', 'AIRL', 'AIRL',
                   'Causal-AIRL', 'Causal-AIRL', 'Causal-AIRL', 'Causal-AIRL', 'Causal-AIRL', 'Causal-AIRL'],
        'Setting': ['Clean (sparse)', 'Clean (shaped)', 'Confounded Z=0 (sparse)', 'Confounded Z=1 (sparse)',
                    'Confounded Z=0 (shaped)', 'Confounded Z=1 (shaped)',
                    'Clean (sparse)', 'Clean (shaped)', 'Confounded Z=0 (sparse)', 'Confounded Z=1 (sparse)',
                    'Confounded Z=0 (shaped)', 'Confounded Z=1 (shaped)'],
        'Reward Type': ['sparse', 'shaped', 'sparse', 'sparse', 'shaped', 'shaped',
                        'sparse', 'shaped', 'sparse', 'sparse', 'shaped', 'shaped'],
        'Reward Spearman': [0.88, 0.86, 0.72, 0.68, 0.74, 0.70, 
                           0.89, 0.88, 0.91, 0.88, 0.90, 0.87],
        'Policy Agreement': [0.93, 0.91, 0.65, 0.60, 0.67, 0.62,
                             0.94, 0.92, 0.87, 0.84, 0.86, 0.83],
        'Cross-Z Agreement': ['—', '—', '65%', '60%', '66%', '61%',
                              '—', '—', '88%', '85%', '87%', '84%']
    }
    df = pd.DataFrame(data)
    
    if reward_type != "all":
        df = df[df['Reward Type'] == reward_type]
    
    return df


def plot_crossz_comparison(reward_type="all"):
    """Plot cross-Z generalization - THE key result."""
    if not VIZ_OK:
        return None
    
    labels = ['Train Z=0\nTest Z=1', 'Train Z=1\nTest Z=0']
    
    # Aggregate or use specific values
    if reward_type == "shaped":
        airl_vals = [0.66, 0.61]
        cairl_vals = [0.87, 0.84]
        title_suffix = "(Shaped Reward)"
    elif reward_type == "sparse":
        airl_vals = [0.65, 0.60]
        cairl_vals = [0.88, 0.85]
        title_suffix = "(Sparse Reward)"
    else:  # all - show average
        airl_vals = [0.655, 0.605]
        cairl_vals = [0.875, 0.845]
        title_suffix = "(All Reward Types)"
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Colorblind-friendly colors
    bars1 = ax.bar(x - width/2, airl_vals, width, label='AIRL', color='#D55E00', edgecolor='black')
    bars2 = ax.bar(x + width/2, cairl_vals, width, label='Causal-AIRL', color='#0072B2', edgecolor='black')
    
    ax.set_ylabel('Policy Agreement', fontsize=12)
    ax.set_xlabel('Cross-Z Evaluation', fontsize=12)
    ax.set_title(f'Cross-Z Generalization {title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 1)
    
    # Reference lines
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=0.7, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
    
    ax.grid(True, axis='y', alpha=0.3)
    
    # Value annotations
    for bars, color in [(bars1, '#8B0000'), (bars2, '#00008B')]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0%}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', va='bottom', fontsize=12, fontweight='bold', color=color)
    
    plt.tight_layout()
    return fig


def plot_method_comparison(reward_type="all"):
    """Plot overall method comparison."""
    if not VIZ_OK:
        return None
    
    metrics = ['Reward\nSpearman', 'Policy\nAgreement', 'Cross-Z\nAgreement']
    
    if reward_type == "shaped":
        airl_vals = [0.74, 0.67, 0.64]
        cairl_vals = [0.90, 0.86, 0.85]
        title_suffix = "(Shaped Reward)"
    elif reward_type == "sparse":
        airl_vals = [0.72, 0.65, 0.63]
        cairl_vals = [0.91, 0.87, 0.86]
        title_suffix = "(Sparse Reward)"
    else:  # all - average
        airl_vals = [0.73, 0.66, 0.635]
        cairl_vals = [0.905, 0.865, 0.855]
        title_suffix = "(All Reward Types)"
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Colorblind-friendly colors
    bars1 = ax.bar(x - width/2, airl_vals, width, label='AIRL', color='#D55E00', edgecolor='black')
    bars2 = ax.bar(x + width/2, cairl_vals, width, label='Causal-AIRL', color='#0072B2', edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Method Comparison {title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Improvement labels
    for i, (a, c) in enumerate(zip(airl_vals, cairl_vals)):
        improvement = c - a
        if improvement > 0:
            ax.annotate(f'+{improvement:.0%}', xy=(i, max(a, c) + 0.05),
                       ha='center', fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN UI
# ============================================================================

# Sidebar filter
st.sidebar.header("🎛️ Filters")
reward_filter = st.sidebar.radio("Reward Type", ["all", "sparse", "shaped"], 
                                  format_func=lambda x: x.capitalize(),
                                  help="Select reward type for all visualisations")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Cross-Z Comparison", 
    "📊 Method Comparison",
    "📋 Results Table",
    "🎯 Key Takeaways"
])

with tab1:
    st.subheader("Cross-Z Generalization: The Key Result")
    st.markdown("""
    **Test:** Train on demos from one expert style (Z=0), evaluate on another (Z=1).
    This tests if the learned reward generalises beyond the specific expert behaviour seen during training.
    """)
    
    if VIZ_OK:
        fig = plot_crossz_comparison(reward_filter)
        if fig:
            st.pyplot(fig)
            plt.close()
    
    # Values based on reward type
    if reward_filter == "shaped":
        airl_crossz, cairl_crossz, improvement = "63.5%", "85.5%", "+22 p.p."
    elif reward_filter == "sparse":
        airl_crossz, cairl_crossz, improvement = "62.5%", "86.5%", "+24 p.p."
    else:
        airl_crossz, cairl_crossz, improvement = "63%", "86%", "+23 p.p."
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AIRL Cross-Z", airl_crossz, delta="-25% vs same-Z", delta_color="inverse")
    with col2:
        st.metric("Causal-AIRL Cross-Z", cairl_crossz, delta="-5% vs same-Z", delta_color="inverse")
    with col3:
        st.metric("Improvement", improvement, delta=f"{improvement}", delta_color="normal")

with tab2:
    st.subheader("Method Comparison (Confounded Setting)")
    
    if VIZ_OK:
        fig = plot_method_comparison(reward_filter)
        if fig:
            st.pyplot(fig)
            plt.close()
    
    reward_label = reward_filter if reward_filter != "all" else "all reward types"
    st.markdown(f"""
    **Setting:** 5×5 ConfoundedGridWorld, {reward_label}, 20 demos  
    **Causal-AIRL advantages:**
    - Higher reward correlation (learns true reward, not expert-style artifacts)
    - Better policy agreement (recovered policy matches expert intent)
    - Much better cross-Z transfer (generalises to unseen expert styles)
    """)

with tab3:
    st.subheader("Experiment Results Summary")
    
    df = get_demo_metrics(reward_filter)
    
    st.dataframe(
        df,
        width='stretch',
        height=400
    )
    
    st.caption("Data from dissertation experiments (sparse and shaped reward conditions)")
    
    csv = df.to_csv(index=False)
    st.download_button("📥 Download as CSV", csv, "causal_airl_results.csv", "text/csv")

with tab4:
    st.subheader("🎯 Key Takeaways")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Main Findings")
        st.markdown("""
        1. **Cross-Z generalization improves by +22-24 p.p.**
           - AIRL: ~63% → Causal-AIRL: ~86%
        
        2. **Reward variance across Z is 5× lower**
           - Learned reward is truly Z-invariant
        
        3. **No performance loss when Z is absent**
           - Causal-AIRL matches AIRL on clean data
        
        4. **Works with both sparse and shaped rewards**
           - Consistent improvements across reward types
        """)
    
    with col2:
        st.markdown("### Why It Works")
        st.markdown("""
        **CausalEncoder** infers latent Z from (s, a) pairs
        
        **Invariance Loss** penalizes:
        ```
        L_inv = Var_z[ R(s) | z ]
        ```
        
        This forces the discriminator to learn a reward that explains expert behaviour **regardless of their style**.
        """)
        
        st.info("📖 See **About** page for architecture details")