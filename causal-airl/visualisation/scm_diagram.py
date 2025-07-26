import os
import matplotlib.pyplot as plt
import networkx as nx


def plot_gridworld_scm(save_path="figures/scm_gridworld.png"):
    """
    SCM for confounded GridWorld:
        Z → π(s)
        Z → T(s'|s,a)
        π(s) → a → s'
    """
    G = nx.DiGraph()

    G.add_edges_from([
        ("Z", "π(s)"),
        ("Z", "T(s'|s,a)"),
        ("π(s)", "a"),
        ("s", "π(s)"),
        ("a", "s'"),
        ("s", "s'"),
        ("T(s'|s,a)", "s'")
    ])

    pos = {
        "Z": (-1, 1),
        "π(s)": (0, 1),
        "a": (1, 1),
        "s": (0, 0),
        "T(s'|s,a)": (1, 0),
        "s'": (2, 0)
    }

    plt.figure(figsize=(6, 3))
    nx.draw(G, pos, with_labels=True, node_color="lightgray", node_size=1600, arrows=True)
    plt.title("SCM: Confounded GridWorld")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[Saved] {save_path}")
    plt.close()


def plot_causal_airl_scm(save_path="figures/scm_causal_airl.png"):
    """
    SCM for Causal-AIRL:
        (s,a) → z → D
        (s,a,s',z) → f → reward
    """
    G = nx.DiGraph()

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
    nx.draw(G, pos, with_labels=True, node_color="lightgray", node_size=1600, arrows=True)
    plt.title("SCM: Causal-AIRL")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[Saved] {save_path}")
    plt.close()


if __name__ == "__main__":
    plot_gridworld_scm()
    plot_causal_airl_scm()
