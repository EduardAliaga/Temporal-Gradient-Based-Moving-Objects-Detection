import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter, uniform_filter


def simple_derivative_filter(frames, index):
    """
    Apply simple 0.5[-1, 0, 1] central difference filter along the temporal axis.
    Computes the temporal derivative at frame[index] using its two neighbors.
    """
    if index == 0 or index >= len(frames) - 1:
        return None
    return 0.5 * (frames[index + 1] - frames[index - 1])


def gaussian_derivative_filter(frames_array, index, sigma):
    """
    Apply 1D derivative of Gaussian along the temporal axis at frame[index].
    """
    margin = int(np.ceil(3 * sigma))
    if index < margin or index >= len(frames_array) - margin:
        return None
    # Extract temporal window centered on target frame
    start = index - margin
    end = index + margin + 1
    window = frames_array[start:end].astype(float)
    # Apply derivative of Gaussian along temporal axis (axis=0)
    deriv_vol = gaussian_filter1d(window, sigma=sigma, axis=0, order=1)
    # Return only the center frame's derivative
    return deriv_vol[margin]


def apply_spatial_smoothing(frames_array, method, sigma_input=None):
    """
    Apply 2D spatial smoothing to each frame independently.
    Supports box filters (3x3, 5x5) and Gaussian with user-defined sigma.
    """
    smoothed = np.zeros_like(frames_array, dtype=float)
    for i in range(len(frames_array)):
        if method == 'box_3x3':
            smoothed[i] = uniform_filter(frames_array[i].astype(float), size=3)
        elif method == 'box_5x5':
            smoothed[i] = uniform_filter(frames_array[i].astype(float), size=5)
        elif method == 'gaussian':
            smoothed[i] = gaussian_filter(frames_array[i].astype(float), sigma=sigma_input)
    return smoothed
