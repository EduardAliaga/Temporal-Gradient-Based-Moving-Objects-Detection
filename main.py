from PIL import Image
import os
import numpy as np

from metrics import compute_derivative_snr, compute_largest_component_ratio
from filters import simple_derivative_filter, gaussian_derivative_filter, apply_spatial_smoothing
from thresholds import threshold_fixed, threshold_percentile, threshold_noise_model
from visualize_temporal_derivatives_only import visualize_temporal_derivatives_only_main, visualize_temporal_derivatives_only_percentile_comparison
from visualize_spatial_temporal_combined import visualize_spatial_temporal_combined_main, visualize_spatial_temporal_combined_percentile_comparison
from visualize_threshold_analysis import visualize_threshold_analysis_fixed_thresholds, visualize_threshold_analysis_noise_model, visualize_threshold_analysis_strategy_comparison
import config

# Load Images
image_dir = config.IMAGE_DIR
grayscale_images = []

for image_file in sorted(os.listdir(image_dir)):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        grayscale_images.append(np.array(image.convert("L"), dtype=float))

frames_array = np.array(grayscale_images)

# Parameters
test_frame_idx = len(grayscale_images) // 2
temporal_sigma_values = config.TEMPORAL_SIGMA_VALUES
spatial_sigma_values = config.SPATIAL_SIGMA_VALUES
percentile_values = config.PERCENTILE_VALUES

# Create results directories
os.makedirs(config.RESULTS_TEMPORAL_DERIVATIVES_ONLY_DIR, exist_ok=True)
os.makedirs(config.RESULTS_SPATIAL_TEMPORAL_COMBINED_DIR, exist_ok=True)
os.makedirs(config.RESULTS_THRESHOLD_ANALYSIS_DIR, exist_ok=True)


# Temporal Derivatives (No Spatial Smoothing)

# Compute all raw derivatives
raw_temporal_derivatives_only = {}

deriv = simple_derivative_filter(grayscale_images, test_frame_idx)
if deriv is not None:
    raw_temporal_derivatives_only['Simple [-1,0,1]'] = {
        'derivative': deriv,
        'frame_range': (test_frame_idx - 1, test_frame_idx + 1),
    }

for t_sigma in temporal_sigma_values:
    deriv = gaussian_derivative_filter(frames_array, test_frame_idx, t_sigma)
    if deriv is not None:
        margin = int(np.ceil(3 * t_sigma))  # margin needed for gaussian filter
        raw_temporal_derivatives_only[f'DoG ts={t_sigma}'] = {
            'derivative': deriv,
            'frame_range': (test_frame_idx - margin, test_frame_idx + margin),
        }

# Build results: each filter x each percentile
results_temporal_derivatives_only = {}
for t_name, t_data in raw_temporal_derivatives_only.items():
    d = t_data['derivative']
    snr = compute_derivative_snr(d)
    for pct in percentile_values:
        mask, thr = threshold_percentile(d, pct)
        mp = 100 * np.sum(mask) / mask.size
        lcc = compute_largest_component_ratio(mask)
        key = f"{t_name} | p={pct}"
        results_temporal_derivatives_only[key] = {
            'derivative': d, 'mask': mask, 'threshold': thr,
            'percentile': pct, 'motion_pct': mp, 'snr': snr, 'lcc': lcc,
            'frame_range': t_data['frame_range'], 'filter_name': t_name,
        }


# Visualize Temporal Derivatives (No Spatial Smoothing)
visualize_temporal_derivatives_only_main(results_temporal_derivatives_only, grayscale_images, test_frame_idx, config.RESULTS_TEMPORAL_DERIVATIVES_ONLY_DIR)
visualize_temporal_derivatives_only_percentile_comparison(raw_temporal_derivatives_only, grayscale_images, test_frame_idx, 
                                     percentile_values, config.RESULTS_TEMPORAL_DERIVATIVES_ONLY_DIR)


# Spatial Smoothing + Temporal Derivatives
spatial_configs = config.SPATIAL_CONFIGS
temporal_configs = config.TEMPORAL_CONFIGS

results_spatial_temporal_combined = {}

for s_name, s_method, s_param in spatial_configs:
    smoothed_frames = apply_spatial_smoothing(frames_array, s_method, s_param)
    smoothed_list = [smoothed_frames[i] for i in range(len(smoothed_frames))]

    for t_name, t_method, t_param in temporal_configs:
        if t_method == 'simple':
            deriv = simple_derivative_filter(smoothed_list, test_frame_idx)
            frame_range = (test_frame_idx - 1, test_frame_idx + 1)
        else:
            deriv = gaussian_derivative_filter(smoothed_frames, test_frame_idx, t_param)
            margin = int(np.ceil(3 * t_param))  # margin for gaussian filter
            frame_range = (test_frame_idx - margin, test_frame_idx + margin)

        if deriv is None:
            continue

        snr = compute_derivative_snr(deriv)

        for pct in percentile_values:
            mask, thr = threshold_percentile(deriv, pct)
            mp = 100 * np.sum(mask) / mask.size
            lcc = compute_largest_component_ratio(mask)

            combo_key = f"{s_name} + {t_name} | p={pct}"
            results_spatial_temporal_combined[combo_key] = {
                'derivative': deriv, 'mask': mask, 'threshold': thr,
                'percentile': pct, 'motion_pct': mp, 'snr': snr, 'lcc': lcc,
                'frame_range': frame_range,
                'spatial': s_name, 'temporal': t_name,
                'test_frame_idx': test_frame_idx,
                'smoothed_frames': smoothed_frames,
            }

# Visualize Spatial Smoothing + Temporal Derivatives
visualize_spatial_temporal_combined_main(results_spatial_temporal_combined, spatial_configs, grayscale_images, test_frame_idx, config.RESULTS_SPATIAL_TEMPORAL_COMBINED_DIR)
visualize_spatial_temporal_combined_percentile_comparison(spatial_configs, temporal_configs, frames_array, 
                                     grayscale_images, test_frame_idx, percentile_values, config.RESULTS_SPATIAL_TEMPORAL_COMBINED_DIR)

# Threshold Analysis & Adaptive Strategy
test_derivatives = {}
test_derivatives['Simple [-1,0,1]'] = simple_derivative_filter(grayscale_images, test_frame_idx)
for t_sigma in temporal_sigma_values:
    d = gaussian_derivative_filter(frames_array, test_frame_idx, t_sigma)
    if d is not None:
        test_derivatives[f'DoG ts={t_sigma}'] = d

# Visualize Threshold Analysis & Adaptive Strategy
visualize_threshold_analysis_fixed_thresholds(test_derivatives, grayscale_images, test_frame_idx, 
                                 config.FIXED_THRESHOLDS, config.RESULTS_THRESHOLD_ANALYSIS_DIR)
visualize_threshold_analysis_noise_model(test_derivatives, grayscale_images, test_frame_idx, 
                           config.K_VALUES, config.RESULTS_THRESHOLD_ANALYSIS_DIR)
visualize_threshold_analysis_strategy_comparison(test_derivatives, grayscale_images, test_frame_idx, 
                                   percentile_values, config.RESULTS_THRESHOLD_ANALYSIS_DIR)


print("TABLE 1")
print(f"{'Filter':<25} {'Pctl':>5} {'Thr':>8} {'Motion%':>9} {'SNR':>8} {'LCC':>8}")
for name, res in results_temporal_derivatives_only.items():
    print(f"{res['filter_name']:<25} {res['percentile']:>5} {res['threshold']:>8.2f} "
          f"{res['motion_pct']:>8.2f}% {res['snr']:>8.2f} {res['lcc']:>8.3f}")

print("TABLE 2")
print(f"{'Spatial':<15} {'Temporal':<20} {'Pctl':>5} {'Thr':>8} {'Motion%':>9} {'SNR':>8} {'LCC':>8}")
for name, res in results_spatial_temporal_combined.items():
    print(f"{res['spatial']:<15} {res['temporal']:<20} {res['percentile']:>5} "
          f"{res['threshold']:>8.2f} {res['motion_pct']:>8.2f}% {res['snr']:>8.2f} {res['lcc']:>8.3f}")

print("THRESHOLD STRATEGY COMPARISON (Simple [-1,0,1])")
print(f"{'Strategy':<15} {'Param':<8} {'Thr':>10} {'Motion%':>10} {'SNR':>10} {'LCC':>10}")

deriv = test_derivatives['Simple [-1,0,1]']
snr_base = compute_derivative_snr(deriv)

for val in [2, 5, 10, 20, 30]:
    mask, thr = threshold_fixed(deriv, val)
    mp = 100 * np.sum(mask) / mask.size
    lcc = compute_largest_component_ratio(mask)
    print(f"{'Fixed':<15} {val:<8} {thr:>10.2f} {mp:>9.2f}% {snr_base:>10.2f} {lcc:>10.3f}")

for pct in percentile_values:
    mask, thr = threshold_percentile(deriv, pct)
    mp = 100 * np.sum(mask) / mask.size
    lcc = compute_largest_component_ratio(mask)
    print(f"{'Percentile':<15} {pct:<8} {thr:>10.2f} {mp:>9.2f}% {snr_base:>10.2f} {lcc:>10.3f}")

for k in [2, 3, 4, 5]:
    mask, thr, _ = threshold_noise_model(deriv, k=k)
    mp = 100 * np.sum(mask) / mask.size
    lcc = compute_largest_component_ratio(mask)
    print(f"{'Noise model':<15} {'k='+str(k):<8} {thr:>10.2f} {mp:>9.2f}% {snr_base:>10.2f} {lcc:>10.3f}")

