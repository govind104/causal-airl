import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


def action_to_vector(action_idx):
    """
    Map action index to vector direction.
    Assumes: 0=up, 1=down, 2=left, 3=right
    """
    return {
        0: (0, -1),
        1: (0, 1),
        2: (-1, 0),
        3: (1, 0)
    }[action_idx]


def plot_policy_vector_field(
    policy: np.ndarray,
    grid_shape: tuple,
    title: str = "Policy Vector Field",
    save_path: str = None,
    terminals=None,
    background=None
):
    H, W = grid_shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)

    for idx, a in enumerate(policy):
        row = idx // W
        col = idx % W
        dx, dy = action_to_vector(int(a))
        U[row, col] = dx
        V[row, col] = dy

    plt.figure(figsize=(5, 4))
    ax = plt.gca()

    if background is not None:
        plt.imshow(background, cmap="gray", alpha=0.3, origin="upper")

    ax.quiver(X, Y, U, V, pivot="middle", color="black", scale=1, scale_units="xy")

    if terminals:
        for (i, j) in terminals:
            plt.text(j, i, '★', ha='center', va='center', color='red', fontsize=14)

    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[Saved] {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--grid_size", type=str, required=True)  # e.g., "(5,5)"
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--title", type=str, default="Policy Vector Field")
    parser.add_argument("--terminals", type=str, default=None)
    parser.add_argument("--background", type=str, default=None)

    args = parser.parse_args()

    policy = np.load(args.policy_path)
    grid_shape = eval(args.grid_size)
    background = np.load(args.background) if args.background else None
    terminals = eval(args.terminals) if args.terminals else None

    plot_policy_vector_field(policy, grid_shape, args.title, args.save_path, terminals, background)
