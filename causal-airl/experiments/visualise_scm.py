import matplotlib.pyplot as plt
import networkx as nx
import os


def plot_gridworld_scm(save_path="results/scm_gridworld.png"):
    """
    SCM for confounded GridWorld:
        Z → π(s)
        Z → T(s'|s,a)
        π(s) → a → s'
    """
    G = nx.DiGraph()

    # Nodes
    G.add_node("Z")
    G.add_node("π(s)")
    G.add_node("a")
    G.add_node("s")
    G.add_node("s'")
    G.add_node("T(s'|s,a)")

    # Edges
    G.add_edges_from([
        ("Z", "π(s)"),
        ("Z", "T(s'|s,a)"),
        ("π(s)", "a"),
        ("s", "π(s)"),
        ("a", "s'"),
        ("s", "s'"),
        ("T(s'|s,a)", "s'")
    ])

    # Layout
    pos = {
        "Z": (-1, 1),
        "π(s)": (0, 1),
        "a": (1, 1),
        "s": (0, 0),
        "T(s'|s,a)": (1, 0),
        "s'": (2, 0)
    }

    plt.figure(figsize=(6, 3))
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=1800, node_color='lightgray')
    plt.title("SCM: Confounded GridWorld")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_causal_airl_scm(save_path="results/scm_causal_airl.png"):
    """
    SCM for Causal-AIRL training:
        (s, a) → z → D
        (s, a, s', z) → f → reward
    """
    G = nx.DiGraph()

    G.add_nodes_from(["s", "a", "s'", "z", "f", "D", "reward"])
    G.add_edges_from([
        ("s", "z"), ("a", "z"),
        ("z", "D"),
        ("s", "f"), ("a", "f"), ("s'", "f"), ("z", "f"),
        ("f", "reward")
    ])

    pos = {
        "s": (0, 1),
        "a": (1, 1),
        "z": (0.5, 0.5),
        "s'": (2, 1),
        "f": (1, 0),
        "D": (0.5, -0.5),
        "reward": (1.5, -0.5)
    }

    plt.figure(figsize=(7, 3))
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=1800, node_color='lightgray')
    plt.title("SCM: Causal-AIRL")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
