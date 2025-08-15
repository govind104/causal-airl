import numpy as np

def bootstrap_mean_ci(x, ci=95, B=1000, rng=None):
    """Percentile bootstrap mean ± CI. Returns (mean, low, high)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(None if rng is None else rng)
    means = []
    n = x.size
    for _ in range(B):
        sample = rng.choice(x, size=n, replace=True)
        means.append(sample.mean())
    low = np.percentile(means, (100-ci)/2)
    high = np.percentile(means, 100-(100-ci)/2)
    return x.mean(), low, high

def cohens_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    n1, n2 = a.size, b.size
    sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2)) if (n1+n2-2) > 0 else np.nan
    return (m1 - m2) / sp if sp and sp > 0 else np.nan
