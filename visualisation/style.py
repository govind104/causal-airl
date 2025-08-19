import matplotlib.pyplot as plt
import matplotlib
import os
from matplotlib import rcParams

def setup_thesis_style():
    """Set up consistent thesis-ready matplotlib styling."""
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True
    })

    # Colorblind-safe palette
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00', '#000000']
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', colors)

def save_figure(fig, path, tight=True):
    """Save figure with consistent settings.

    Args:
        fig: Matplotlib figure object
        path: Output file path
        tight: Whether to call fig.tight_layout(). If the figure uses
               constrained_layout, tight_layout is skipped automatically.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # If the figure uses constrained_layout, skip tight_layout to avoid warnings
    use_tight = bool(tight)
    try:
        if hasattr(fig, "get_constrained_layout") and fig.get_constrained_layout():
            use_tight = False
    except Exception:
        pass
    if use_tight:
        fig.tight_layout()

    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)
