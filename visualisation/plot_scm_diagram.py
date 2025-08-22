import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from numpy import array, linalg

# Embed TrueType for crisp PDF text
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['text.usetex'] = False

def shrink_node(p1, p2, shape1='o', shape2='o'):
    # Approximate radii for circle vs square
    shape_radius = {'o': 0.25, 's': 0.30}
    r1 = shape_radius.get(shape1, 0.25)
    r2 = shape_radius.get(shape2, 0.25)

    v = array(p2) - array(p1)
    v = v / linalg.norm(v)
    return tuple(array(p1) + v * r1), tuple(array(p2) - v * r2)

def draw_custom_scm(method, G, pos, latent_nodes, edge_colors, edge_styles,
                    title, save_path, special_nodes=None, edge_curves=None,
                    x_span=None, stage_bands=None):
    """
    special_nodes: dict like {
        'nodes': ['V(s)', "V(s')"],
        'node_size': 1200,
        'node_color': '#eeeeee',
        'edgecolors': 'gray',
        'linewidths': 1.0,
        'node_shape': 'o',
        'label_overrides': {node: r'$V(s)$', ...}
    }
    """
    # Slightly wider canvas to accommodate a strict 5×2 / 4×2 grid
    plt.figure(figsize=(10, 5.6))
    ax = plt.gca()
    ax.set_title(title, pad=30, fontsize=12)
    ax.set_ylim(-1.5, 1.5)
    if x_span:
        ax.set_xlim(x_span[0], x_span[1])
    edge_curves = edge_curves or {}

    # Node styling. Avoid duplicate drawing of "special" nodes (e.g., V(s), V(s'))
    special_list = special_nodes.get('nodes', []) if (special_nodes and special_nodes.get('nodes')) else []
    observed_nodes = [n for n in G.nodes if n not in latent_nodes and n not in special_list]
    latent_nodes_base = [n for n in latent_nodes if n not in special_list]
    node_shapes = {n: ('s' if n in latent_nodes else 'o') for n in G.nodes}

    # Base nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=observed_nodes, node_color='lightgray', node_size=1800
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=latent_nodes_base, node_color='white', node_size=1800,
        edgecolors='black', linewidths=2, node_shape='s'
    )

    # Edges with precise arrow endpoints (account for node shapes)
    for (u, v) in G.edges:
        color = edge_colors.get((u, v), 'black')
        style = edge_styles.get((u, v), 'solid')
        shape_u = node_shapes.get(u, 'o')
        shape_v = node_shapes.get(v, 'o')
        pA, pB = shrink_node(pos[u], pos[v], shape_u, shape_v)
        arrow = FancyArrowPatch(
            posA=pA, posB=pB, arrowstyle='->', color=color, linestyle=style,
            mutation_scale=16, linewidth=2,
            connectionstyle=f"arc3,rad={edge_curves.get((u, v), 0.0)}",
            transform=ax.transData
        )
        arrow.set_zorder(2)
        ax.add_patch(arrow)

    # Labels (math font) — exclude special nodes so they aren't labeled twice
    base_label_nodes = [n for n in G.nodes if n not in special_list]
    math_labels = {n: n for n in base_label_nodes}
    nx.draw_networkx_labels(G, pos, labels=math_labels, font_size=9, font_family="DejaVu Sans")

    # Overlay special/smaller nodes (e.g., V(s), V(s'))
    legend_extra = []
    if special_nodes and special_nodes.get('nodes'):
        nlist = special_nodes['nodes']
        nx.draw_networkx_nodes(
            G, pos, nodelist=nlist,
            node_color=special_nodes.get('node_color', '#eeeeee'),
            node_size=special_nodes.get('node_size', 1200),
            edgecolors=special_nodes.get('edgecolors', 'gray'),
            linewidths=special_nodes.get('linewidths', 1.0),
            node_shape=special_nodes.get('node_shape', 'o'),
        )
        # Optional label overrides
        if 'label_overrides' in special_nodes:
            nx.draw_networkx_labels(
                G, pos,
                labels=special_nodes['label_overrides'],
                font_size=9, font_family="DejaVu Sans"
            )
        # Legend marker for potentials
        legend_extra.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       label=r'Potential terms: $V(s),\,V(s^\prime)$',
                       markerfacecolor='#eeeeee', markeredgecolor='gray',
                       markersize=10)
        )

    # Stage bands + legend
    if method == "ConfoundedGW":
        # 5 bars: Context/Input, Policy, Dynamics, Action, Outcome
        for x, label in stage_bands:
            ax.axvspan(x - 0.7, x + 0.7, color='lightblue', alpha=0.12, zorder=0)
            ax.text(x, 1.25, label, ha='center', va='center', fontsize=9, alpha=0.7)

        legend_elements = [
            plt.Line2D([0], [0], color='black', lw=2, label='Environment Dynamics'),
            plt.Line2D([0], [0], color='blue',  lw=2, label='Expert Policy Input'),
            plt.Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='Latent Influence'),
            plt.Line2D([0], [0], marker='s', color='w',
                       label=r'Expert Policy: $\pi_E(a\mid s,z)$',
                       markerfacecolor='#99ccff', markeredgecolor='blue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w',
                       label=r'Latent Variable: $Z$ (Unobserved)',
                       markerfacecolor='white', markeredgecolor='black', markersize=10),
        ]
    elif method == "Causal-AIRL":
        # 4 bars: Inputs, Latent Variable, Feature+Discriminator, Output
        for x, label in stage_bands:
            ax.axvspan(x - 0.7, x + 0.7, color='lightgreen', alpha=0.12, zorder=0)
            ax.text(x, 1.25, label, ha='center', va='center', fontsize=9, alpha=0.7)

        legend_elements = [
            plt.Line2D([0], [0], color='black', lw=2, label='Environment Dynamics'),
            plt.Line2D([0], [0], color='blue',  lw=2, label='Expert Policy Input'),
            plt.Line2D([0], [0], color='red',   lw=2, label='Learner Signal'),
            plt.Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='Latent Influence'),
            plt.Line2D([0], [0], marker='o', color='w',
                       label=r'Feature Extractor: $\phi(s,a,z)$',
                       markerfacecolor='lightgray', markersize=10),
            plt.Line2D([0], [0], marker='^', color='w',
                       label=r'Discriminator: $D$',
                       markerfacecolor='#ffcccc', markeredgecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w',
                       label=r'Latent Variable: $Z$ (Inferred)',
                       markerfacecolor='white', markeredgecolor='black', markersize=10),
        ] + legend_extra
    else:
        raise ValueError(f"Unknown method: {method}")

    legend = ax.legend(
        handles=[h for h in legend_elements if h is not None],
        loc='lower center', bbox_to_anchor=(0.5, -0.27), ncol=3, fontsize=9,
        handletextpad=0.6, columnspacing=2.0, labelspacing=1.2
    )

    plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', bbox_extra_artists=[legend], pad_inches=0.2)
    plt.savefig(save_path.replace(".png", ".pdf"), format='pdf', bbox_inches='tight',
                bbox_extra_artists=[legend], pad_inches=0.2)
    base, _ = os.path.splitext(save_path)
    print(f"[Saved] {save_path} and {base}.pdf")
    plt.close()

def plot_gridworld_scm():
    """
    SCM: Confounded GridWorld (publication-ready)
    Required arrows: Z→S, Z→A, and (S,A)→S' via T(s'|s,a).
    No Z→T edge (do not imply T depends on Z).
    """
    G = nx.DiGraph()

    # --- Nodes (mathtext) ---
    Z   = r'$Z$'
    s   = r'$s$'
    piE = r'$\pi_E(a\mid s,z)$'
    Tsa = r"$T(s' \mid s,a)$"
    a   = r'$a$'
    sp  = r"$s'$"

    # Edges
    G.add_edges_from([
        (Z,  s),          # Z → S (latent context affects state)
        (Z,  a),          # Z → A (latent context affects action)
        (s,  piE),        # S → π_E
        (piE, a),         # π_E → A
        (s,  Tsa),        # S → T
        (a,  Tsa),        # A → T
        (Tsa, sp),        # T → S'
    ])

    # Columnar grid (5×2): x0..x4 centered in each bar
    GW_X = [0.0, 1.6, 3.2, 4.8, 6.4]
    y_top, y_bot = 0.75, -0.75
    pos = {
        # x0
        Z:   (GW_X[0], y_top),    # Z at Context/Input (top)
        s:   (GW_X[0], y_bot),    # s at Context/Input (bottom)
        # x1
        piE: (GW_X[1], y_top),    # π_E at Policy (top)
        # x2
        Tsa: (GW_X[2], y_bot),    # T at Dynamics (bottom)
        # x3
        a:   (GW_X[3], y_top),    # a at Action (top)
        # x4
        sp:  (GW_X[4], y_top),    # s' at Outcome (top)
    }

    latent_nodes = [Z]
    latent_edges = [(Z, s), (Z, a)]

    edge_colors = {
        (Z, s): 'blue',
        (Z, a): 'blue',
        (s, piE): 'black',
        (piE, a): 'blue',
        (s, Tsa): 'black',
        (a, Tsa): 'black',
        (Tsa, sp): 'black',
    }
    edge_styles = {e: 'dashed' for e in latent_edges}

    # Gentle curvature to avoid kissing borders & crossings
    edge_curves = {
        (Z, s):  0.15,
        (Z, a): -0.15,
        (s, piE): 0.04,
       (piE, a): 0.04,
        (s, Tsa): 0.06,
        (a, Tsa): -0.06,
        (Tsa, sp): 0.02,
    }
    # Bars (centers) for background spans and titles
    stage_centers = [
        (GW_X[0], "Context / Input"),
        (GW_X[1], "Policy"),
        (GW_X[2], "Dynamics"),
        (GW_X[3], "Action"),
        (GW_X[4], "Outcome"),
    ]


    draw_custom_scm(
        method="ConfoundedGW",
        G=G, pos=pos, latent_nodes=latent_nodes,
        edge_colors=edge_colors, edge_styles=edge_styles, edge_curves=edge_curves,
        title="SCM: Confounded GridWorld",
        save_path="results/figures/scm_gridworld_confounded.png",
        x_span=(GW_X[0] - 0.8, GW_X[-1] + 0.8),
        stage_bands=stage_centers
    )

def plot_causal_airl_scm():
    """
    SCM: Causal-AIRL with potential terms to reflect f_{θ,ψ} decomposition.
    Inputs (s,a,s') + inferred z → φ(s,a,z), D; small grey V(s), V(s') → D.
    """
    G = nx.DiGraph()

    # --- Nodes (mathtext) ---
    s    = r'$s$'
    a    = r'$a$'
    sp   = r"$s'$"
    z    = r'$z$'
    phi  = r'$\phi(s,a,z)$'
    D    = r'$D$'
    R    = r'$r_\theta$'
    Vs   = r'$V(s)$'
    Vsp  = r"$V(s')$"

    # Edges
    G.add_edges_from([
        (s,  phi), (a,  phi), (z, phi),              # to features (no s' → φ)
        (phi, R),                                    # feature → reward/output
        (s,  D), (a, D), (z, D),                     # inputs → discriminator
        (Vs, D), (Vsp, D),                           # potential terms → discriminator
    ])

    # Columnar grid (4×2): x0..x3 centered in each bar
    CAIRL_X = [0.0, 2.2, 4.4, 6.6]
    y_top, y_bot = 0.75, -0.75
    pos = {
        # x0 (Inputs)
        s:   (CAIRL_X[0], y_top),
        a:   (CAIRL_X[0], y_bot),
        # x1 (Latent)
        z:   (CAIRL_X[1], y_bot),
        # x2 (Feature + Discriminator)
        phi: (CAIRL_X[2], y_top),
        D:   (CAIRL_X[2], y_bot),
        # x3 (Output)
        R:   (CAIRL_X[3], y_top),
        sp:  (CAIRL_X[3], y_bot),
        # Potentials hugging D
        Vs:  (CAIRL_X[2]-0.7, -0.25),
        Vsp: (CAIRL_X[2]-0.7, -1.25),
    }

    latent_nodes = [z]
    latent_edges = [(z, phi), (z, D)]

    edge_colors = {
        (z, D): 'black',     # keep latent influence dashed (style below)
        (phi, R): 'red',     # learner/output signal if you want emphasis
        (z, phi): 'black',
        (s, phi): 'black',
        (a, phi): 'black',
        (s, D): 'black',
        (a, D): 'black',
        (Vs, D): 'black',
        (Vsp, D): 'black',
    }
    edge_styles = {e: 'dashed' for e in latent_edges}

    # Staggered curvature to keep feeds distinct into φ and D
    edge_curves = {
        (s, phi):  0.06, (a, phi): -0.06, (z, phi): 0.00,
        (s, D):    0.10,
        (a, D):   0.22,
        (z, D):  -0.10,
        (Vs, D):   0.12, (Vsp, D):-0.12,
        (phi, R):  0.04,
    }

    # Special smaller nodes for potential terms
    special_nodes = {
        'nodes': [Vs, Vsp],
        'node_size': 1350,
        'node_color': '#eeeeee',
        'edgecolors': 'gray',
        'linewidths': 1.0,
        'node_shape': 'o',
        'label_overrides': {Vs: r'$V(s)$', Vsp: r'$V(s^\prime)$'},
    }

    draw_custom_scm(
        method="Causal-AIRL",
        G=G, pos=pos, latent_nodes=latent_nodes,
        edge_colors=edge_colors, edge_styles=edge_styles, edge_curves=edge_curves,
        title="SCM: Causal-AIRL",
        save_path="results/figures/scm_airl_learner.png",
        special_nodes=special_nodes,
        x_span=(CAIRL_X[0]-0.8, CAIRL_X[-1]+0.8),
        stage_bands=[
            (CAIRL_X[0], "Inputs"),
            (CAIRL_X[1], "Latent Variable"),
            (CAIRL_X[2], "Feature + Discriminator"),
            (CAIRL_X[3], "Output"),
        ]
    )

if __name__ == "__main__":
    plot_gridworld_scm()
    plot_causal_airl_scm()
