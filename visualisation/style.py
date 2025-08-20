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
    # Improve legend legibility (subtle frame with slight opacity)
    plt.rcParams.update({'legend.framealpha': 0.9, 'legend.edgecolor': '#dddddd'})

def save_figure(fig, path, tight=True):
    """Save figure with consistent settings.

    Args:
        fig: Matplotlib figure object
        path: Output file pathW
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

# Canonical method ordering and labels (used across figures)
METHOD_ORDER = ['ng', 'maxent', 'airl', 'causal_airl']
METHOD_LABELS = {
    'ng': 'Ng-Russell',
    'maxent': 'MaxEnt',
    'airl': 'AIRL',
    'causal_airl': 'Causal-AIRL',
    # tolerate variants seen in configs
    'NG': 'Ng-Russell',
    'MAXENT': 'MaxEnt',
    'AIRL': 'AIRL',
    'CAUSAL_AIRL': 'Causal-AIRL',
}

def method_label(m):
    return METHOD_LABELS.get(str(m), str(m))

def set_method_color_cycle(methods):
    """Set a stable, method-aware color order subset of the global palette."""
    # Use the global color cycle as a palette source
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    # Map methods to indices by METHOD_ORDER
    order = [m for m in METHOD_ORDER if m in methods] + [m for m in methods if m not in METHOD_ORDER]
    colors = {}
    for i, m in enumerate(order):
        colors[m] = prop_cycle[i % len(prop_cycle)]
    # Build cycle in the requested 'methods' order
    cycle = [colors[m] for m in methods]
    plt.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=cycle)
