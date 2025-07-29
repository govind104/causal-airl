import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from numpy import array, linalg
mpl.rcParams['pdf.fonttype'] = 42  # TrueType font for PDF
mpl.rcParams['ps.fonttype'] = 42

def draw_custom_scm(method, G, pos, latent_nodes, edge_colors, edge_styles, title, save_path):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_title(title, pad=30, fontsize=12)
    ax.set_ylim(-1.5, 1.5)

    # Draw all nodes
    observed_nodes = [n for n in G.nodes if n not in latent_nodes]
    node_shapes = {n: ('s' if n in latent_nodes else 'o') for n in G.nodes}
    nx.draw_networkx_nodes(G, pos, nodelist=observed_nodes, node_color='lightgray', node_size=2000)
    nx.draw_networkx_nodes(G, pos, nodelist=latent_nodes, node_color='white', node_size=2000,
                           edgecolors='black', linewidths=2, node_shape='s')

    # Draw all edges
    for (u, v) in G.edges:
        color = edge_colors.get((u, v), 'black')
        style = edge_styles.get((u, v), 'solid')
        shape_u = node_shapes[u]
        shape_v = node_shapes[v]
        pA, pB = shrink_node(pos[u], pos[v], shape_u, shape_v)
        arrow = FancyArrowPatch(posA=pA, posB=pB, arrowstyle='->', color=color, linestyle=style, 
                                mutation_scale=15, linewidth=2, connectionstyle="arc3,rad=0.0", transform=ax.transData)
        arrow.set_zorder(5)
        ax.add_patch(arrow)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")

    # Add vertical bands for each stage
    if method == "ConfoundedGW":
        stage_bands = [
            (0, "Context / Input"),       # s and Z
            (1.5, "Policy / Dynamics"),     # π and T
            (3, "Action"),
            (4.5, "Outcome"),
        ]
        for x, label in stage_bands:
            ax.axvspan(x - 0.675, x + 0.675, color='lightblue', alpha=0.12, zorder=0)
            ax.text(x, 1.25, label, ha='center', va='center', fontsize=9, alpha=0.7)
        # Draw legend
        legend_elements = [
            # Group 1: Edge types
            plt.Line2D([0], [0], color='black', lw=2, label='Environment Dynamics'),
            plt.Line2D([0], [0], color='blue', lw=2, label='Expert Policy Input'),
            plt.Line2D([0], [0], color='red', lw=2, label='Learner Signal'),
            plt.Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='Latent Influence'),
            # Group 2: Node types
            plt.Line2D([0], [0], marker='s', color='w', label='Expert Policy: π(s)', markerfacecolor='#99ccff', markeredgecolor='blue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Latent Variable: Z (Unobserved)', markerfacecolor='white', markeredgecolor='black', markersize=10)
        ]

    elif method == "Causal-AIRL":
        stage_bands = [
            (0, "Inputs"),              # s, a
            (1.5, "Latent Variable"),   # Z
            (3, "Feature + Discriminator"), 
            (4.5, "Output"),
        ]
        for x, label in stage_bands:
            ax.axvspan(x - 0.675, x + 0.675, color='lightgreen', alpha=0.12, zorder=0)
            ax.text(x, 1.25, label, ha='center', va='center', fontsize=9, alpha=0.7)
        # Draw legend
        legend_elements = [
            # Group 1: Edge types
            plt.Line2D([0], [0], color='black', lw=2, label='Environment Dynamics'),
            plt.Line2D([0], [0], color='blue', lw=2, label='Expert Policy Input'),
            plt.Line2D([0], [0], color='red', lw=2, label='Learner Signal'),
            plt.Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='Latent Influence'),
            # Group 2: Node types
            plt.Line2D([0], [0], marker='o', color='w', label='Feature Extractor: ϕ(s,a,Z)', markerfacecolor='lightgray', markersize=10),
            plt.Line2D([0], [0], marker='^', color='w', label='Discriminator: D', markerfacecolor='#ffcccc', markeredgecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Latent Variable: Z (Inferred)', markerfacecolor='white', markeredgecolor='black', markersize=10),
        ]
    else:
        print(f"{method} method not defined.")

    legend = ax.legend(handles=[h for h in legend_elements if h is not None], loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=9, 
                       handletextpad=0.6, columnspacing=2.0, labelspacing=1.2)

    plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', bbox_extra_artists=[legend], pad_inches=0.2)
    plt.savefig(save_path.replace(".png", ".pdf"), format='pdf', bbox_inches='tight', bbox_extra_artists=[legend], pad_inches=0.2)
    base, _ = os.path.splitext(save_path)
    print(f"[Saved] {save_path} and {base}.pdf")    
    plt.close()

def shrink_node(p1, p2, shape1='o', shape2='o'):
    # Define radii based on node shape
    shape_radius = {'o': 0.25, 's': 0.3}
    r1 = shape_radius.get(shape1, 0.25)
    r2 = shape_radius.get(shape2, 0.25)
    
    v = array(p2) - array(p1)
    v = v / linalg.norm(v)
    return tuple(array(p1) + v * r1), tuple(array(p2) - v * r2)

def plot_gridworld_scm():
    """
    SCM for confounded GridWorld:
        Z → π(s)
        Z → T(s'|s,a)
        π(s) → a → s'
    """
    G = nx.DiGraph()

    # Edges
    G.add_edges_from([
        ("Z", "π(s)"),  # Confounder influences policy
        ("Z", "T(s'|s,a)"),              # Confounder influences transition dynamics
        ("π(s)", "a"),                # Expert policy outputs action
        ("s", "π(s)"),                # State informs expert policy
        ("a", "s'"),                                   # Action affects next state
        ("T(s'|s,a)", "s'")                            # Transition model governs dynamics
    ])

    # Layered left-to-right positions for gridworld SCM
    pos = {
        "Z":    (0, 0.75),     # Confounder
        "s":                  (0, -0.75),    # State

        "π(s)": (1.5, 0.75),  # Policy
        "T(s'|s,a)":          (1.5, -0.75),    # Transition model

        "a":                  (3, 0.75),   # Action

        "s'":                 (4.5, 0.75),     # Next state
    }

    latent_nodes = ["Z"]
    latent_edges = [("Z", "π(s)"), ("Z", "T(s'|s,a)")]
    edge_colors = {
        ("Z", "π(s)"): 'blue',
        ("Z", "T(s'|s,a)"): 'blue',
        ("π(s)", "a"): 'blue',
        ("T(s'|s,a)", "s'"): 'black',
        ("s", "π(s)"): 'black',
        ("a", "s'"): 'black',
        ("s", "s'"): 'black'
    }
    edge_styles = {e: 'dashed' for e in latent_edges}

    method = "ConfoundedGW"
    save_path = "results/figures/scm_gridworld_confounded.png"
    draw_custom_scm(method, G, pos, latent_nodes, edge_colors, edge_styles,
                    "SCM: Confounded GridWorld", save_path)
    
def plot_causal_airl_scm():
    """
    SCM for Causal-AIRL:
        (s,a) → Z → D
        (s,a,s',Z) → f → Reward
    """
    G = nx.DiGraph()

    # Edges
    G.add_edges_from([
        ("s", "Z"),
        ("a", "Z"),
        ("Z", "D"),
        ("s", "ϕ(s,a,Z)"),
        ("a", "ϕ(s,a,Z)"),
        ("s'", "ϕ(s,a,Z)"),
        ("Z", "ϕ(s,a,Z)"),
        ("ϕ(s,a,Z)", "Reward")
    ])

    # Layered left-to-right positions for Causal-AIRL SCM
    pos = {
        "s":                             (0, 0.75),    # Inputs
        "a":                             (0, -0.75),

        "Z":                             (1.5, -0.75),  # Latent confounder

        "ϕ(s,a,Z)": (3, 0.75),  # Feature extractor
        "D":            (3, -0.75),

        "Reward":                        (4.5, 0.75),  # Output
        "s'":                            (4.5, -0.75), # Next state
    }

    latent_nodes = ["Z"]
    latent_edges = [("Z", "D"), ("Z", "ϕ(s,a,Z)")]
    edge_colors = {
        ("Z", "D"): 'red',
        ("ϕ(s,a,Z)", "Reward"): 'red',
        ("Z", "ϕ(s,a,Z)"): 'black',
        ("s", "ϕ(s,a,Z)"): 'black',
        ("a", "ϕ(s,a,Z)"): 'black',
        ("s'", "ϕ(s,a,Z)"): 'black',
    }
    edge_styles = {e: 'dashed' for e in latent_edges}

    method = "Causal-AIRL"
    save_path = "results/figures/scm_airl_learner.png"
    draw_custom_scm(method, G, pos, latent_nodes, edge_colors, edge_styles,
                    "SCM: Causal-AIRL", save_path)

if __name__ == "__main__":
    plot_gridworld_scm()
    plot_causal_airl_scm()