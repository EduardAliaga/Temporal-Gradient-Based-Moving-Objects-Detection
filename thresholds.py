import numpy as np


def threshold_fixed(derivative, value):
    """
    Apply a fixed threshold to the absolute temporal derivative.
    """
    abs_deriv = np.abs(derivative)
    mask = (abs_deriv > value).astype(np.uint8)
    return mask, value


def threshold_percentile(derivative, percentile):
    """
    Threshold based on a percentile of the absolute derivative values.
    """
    abs_deriv = np.abs(derivative)
    thr = np.percentile(abs_deriv, percentile)
    mask = (abs_deriv > thr).astype(np.uint8)
    return mask, thr


def threshold_noise_model(derivative, k=3.0):
    """
    Adaptive threshold based on modeling background derivatives as Gaussian noise.
    """
    abs_deriv = np.abs(derivative)
    median_val = np.median(abs_deriv)
    mad = np.median(np.abs(abs_deriv - median_val))
    sigma_noise = mad / 0.6745
    thr = k * sigma_noise
    mask = (abs_deriv > thr).astype(np.uint8)
    return mask, thr, sigma_noise