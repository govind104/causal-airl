"""
🎮 Interactive Demo - Causal-AIRL Portfolio
=============================================
TRULY interactive: live trajectory simulation, adjustable parameters,
visual demonstration of confounding effects.
"""

import streamlit as st
import numpy as np
import os
import sys
import time

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

st.set_page_config(page_title="Interactive Demo", page_icon="🎮", layout="wide")

st.title("🎮 Interactive Demo")
st.markdown("Explore the GridWorld environment and see how confounding affects learning.")

# Safe matplotlib import
VIZ_OK = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    VIZ_OK = True
except ImportError:
    st.error("matplotlib required. Install with `pip install matplotlib`")


def generate_reward(grid_size, reward_type):
    """Generate ground truth reward."""
    H, W = grid_size
    if reward_type == "sparse":
        reward = np.zeros((H, W))
        reward[H-1, W-1] = 1.0
    else:
        reward = np.zeros((H, W))
        goal = (H-1, W-1)
        for i in range(H):
            for j in range(W):
                reward[i, j] = -(abs(i - goal[0]) + abs(j - goal[1]))
    return reward


def simulate_trajectory(grid_size, z_value, slip_prob=0.0):
    """Simulate an expert trajectory with confounding."""
    H, W = grid_size
    pos = (0, 0)
    goal = (H-1, W-1)
    trajectory = [pos]
    
    max_steps = H * W * 2
    for _ in range(max_steps):
        if pos == goal:
            break
        
        # Confounded policy
        if z_value == 0:
            # Z=0: Optimal goal-seeking
            if pos[0] < goal[0]:
                action = 1  # DOWN
            else:
                action = 3  # RIGHT
        else:
            # Z=1: Mixed style - sometimes takes detours
            if np.random.rand() < 0.3:
                action = np.random.choice([0, 2])  # UP or LEFT sometimes
            elif pos[0] < goal[0]:
                action = 1
            else:
                action = 3
        
        # Slip
        if np.random.rand() < slip_prob:
            action = np.random.randint(4)
        
        # Execute
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        di, dj = deltas[action]
        new_pos = (max(0, min(H-1, pos[0] + di)), max(0, min(W-1, pos[1] + dj)))
        pos = new_pos
        trajectory.append(pos)
    
    return trajectory


def plot_gridworld_with_trajectory(grid_size, reward, trajectories, title):
    """Plot GridWorld with trajectory overlay."""
    if not VIZ_OK:
        return None
    
    H, W = grid_size
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Use appropriate colormap based on reward type
    if reward.max() - reward.min() < 0.5:  # Sparse reward (mostly zeros)
        # Light gray background with just goal highlighted
        ax.set_facecolor('#fafafa')
        im = ax.imshow(reward, cmap='Greens', origin='upper', alpha=0.6, vmin=0, vmax=1)
    else:
        # Shaped reward - use diverging colormap
        im = ax.imshow(reward, cmap='RdBu_r', origin='upper', alpha=0.7)
    
    # Grid lines
    for i in range(H + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    for j in range(W + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    
    # Trajectories - using colorblind-friendly palette (tab10)
    # Colors: blue, orange, green, purple, brown - all distinguishable
    cb_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#7f7f7f', 
                '#bcbd22', '#17becf', '#d62728']
    for i, traj in enumerate(trajectories):
        color = cb_colors[i % len(cb_colors)]
        coords = np.array([(p[1], p[0]) for p in traj])  # (x, y)
        ax.plot(coords[:, 0], coords[:, 1], '-', color=color, linewidth=2.5, alpha=0.8)
        ax.plot(coords[0, 0], coords[0, 1], 'o', color=color, markersize=10, markeredgecolor='white', markeredgewidth=1)
        ax.plot(coords[-1, 0], coords[-1, 1], 's', color=color, markersize=10, markeredgecolor='white', markeredgewidth=1)
    
    # Goal
    ax.plot(W-1, H-1, 'r*', markersize=25, markeredgecolor='black', zorder=10)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.axis('off')
    plt.tight_layout()
    return fig


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

st.sidebar.header("🎛️ Environment")

grid_size = st.sidebar.select_slider(
    "Grid Size",
    options=[3, 5, 7],
    value=5,
    format_func=lambda x: f"{x}×{x}"
)
grid_size = (grid_size, grid_size)

reward_type = st.sidebar.radio(
    "Reward Type",
    options=["sparse", "shaped"],
    help="Sparse: +1 at goal only. Shaped: -distance gradient."
)

slip_prob = st.sidebar.slider(
    "Slip Probability",
    0.0, 0.3, 0.0, 0.05,
    help="Probability of random action"
)

st.sidebar.divider()
st.sidebar.header("🔀 Confounding")

z_value = st.sidebar.radio(
    "Expert Style (Z)",
    options=[0, 1],
    format_func=lambda z: f"Z={z}: {'Optimal path' if z == 0 else 'Mixed/suboptimal'}",
    help="Latent confounder affecting expert behaviour"
)

num_demos = st.sidebar.slider("Number of demos", 3, 10, 5)


# ============================================================================
# MAIN CONTENT
# ============================================================================

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🎲 Live Trajectory Simulation")
    
    if st.button("🔄 Simulate New Demos", type="primary"):
        st.session_state['trajectories'] = [
            simulate_trajectory(grid_size, z_value, slip_prob) 
            for _ in range(num_demos)
        ]
    
    # Initialize if needed
    if 'trajectories' not in st.session_state:
        st.session_state['trajectories'] = [
            simulate_trajectory(grid_size, z_value, slip_prob) 
            for _ in range(num_demos)
        ]
    
    reward = generate_reward(grid_size, reward_type)
    trajectories = st.session_state['trajectories']
    
    if VIZ_OK:
        fig = plot_gridworld_with_trajectory(
            grid_size, reward, trajectories,
            f"Expert Demos (Z={z_value}, {reward_type} reward)"
        )
        if fig:
            st.pyplot(fig)
            plt.close()
    
    # Stats
    lengths = [len(t) for t in trajectories]
    H, W = grid_size
    goal_reaches = sum(1 for t in trajectories if t[-1] == (H-1, W-1))
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Avg Steps", f"{np.mean(lengths):.1f}")
    col_b.metric("Goal Rate", f"{goal_reaches}/{len(trajectories)}")
    col_c.metric("Z Value", z_value)

with col2:
    st.subheader("🔍 What This Shows")
    
    if z_value == 0:
        st.success("""
        **Z=0: Optimal Expert**
        
        Trajectories go directly to goal. This is the "clean" expert style.
        
        Both AIRL and Causal-AIRL learn well from these demos.
        """)
    else:
        st.warning("""
        **Z=1: Suboptimal/Mixed Expert**
        
        Trajectories sometimes take detours or suboptimal paths.
        
        - **AIRL** overfits to this style → poor cross-Z transfer
        - **Causal-AIRL** learns Z-invariant reward → generalises
        """)
    
    st.divider()
    
    st.markdown("### The Cross-Z Test")
    st.markdown("""
    **Key experiment:** Train on Z=0, test on Z=1 (and vice versa).
    
    | Method | Cross-Z Agreement |
    |--------|------------------|
    | AIRL | ~65% |
    | **Causal-AIRL** | **~85%** |
    
    *+20 percentage point improvement!*
    """)
    
    st.info("👉 See **Metrics Dashboard** for the full comparison chart.")

st.divider()

# Problem/Solution explanation
st.subheader("🎯 Why This Matters")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1️⃣ The Problem")
    st.markdown("""
    Real experts have **unobserved preferences** (Z):
    - Risk tolerance
    - Expertise level  
    - Contextual knowledge
    
    Standard IRL conflates reward with style.
    """)

with col2:
    st.markdown("### 2️⃣ The Solution")
    st.markdown("""
    **Causal-AIRL** infers Z from demos and learns a reward that is **invariant to Z**.
    
    Key component: **Invariance Loss**
    ```
    L_inv = Var_z[R(s) | z]
    ```
    """)

with col3:
    st.markdown("### 3️⃣ The Result")
    st.markdown("""
    The learned reward captures **what** the expert wants, not **how** they achieve it.
    
    ✅ Generalises to new expert styles  
    ✅ 5× lower reward variance  
    ✅ Same performance when no confounding
    """)
