import numpy as np
from scipy.ndimage import label


def compute_derivative_snr(derivative):
    """
    Estimate signal-to-noise ratio of the temporal derivative.
    """
    abs_d = np.abs(derivative).ravel()
    fg_vals = abs_d[abs_d >= np.percentile(abs_d, 95)]
    bg_vals = abs_d[abs_d <= np.percentile(abs_d, 50)]
    if len(bg_vals) == 0 or np.std(bg_vals) == 0:
        return 0.0
    return np.mean(fg_vals) / (np.std(bg_vals) + 1e-8)


def compute_largest_component_ratio(mask):
    """
    Measure spatial coherence of the motion mask.
    """
    labeled, num = label(mask)
    if num == 0:
        return 0.0
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background label
    total = np.sum(mask)
    if total == 0:
        return 0.0
    return sizes.max() / total

