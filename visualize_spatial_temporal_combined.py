import numpy as np
import matplotlib.pyplot as plt
from filters import apply_spatial_smoothing, simple_derivative_filter, gaussian_derivative_filter
from thresholds import threshold_percentile


def visualize_spatial_temporal_combined_main(results_spatial_temporal_combined, spatial_configs, grayscale_images, test_frame_idx, output_dir):
    """
    Visualize Spatial Smoothing + Temporal Derivatives results: one figure per spatial filter.
    Each row shows a temporal filter with 5 columns: smoothed frame before, smoothed frame after, derivative magnitude, binary mask, overlay.
    Only the 90th percentile results are shown to keep figures compact.
    """
    for s_name, s_method, s_param in spatial_configs:
        safe_name = s_name.replace(' ', '_').replace('=', '')
        show_only_p90 = safe_name in ['Box_3x3', 'Box_5x5', 'Gauss_ss0.5', 'Gauss_ss1.5', 'Gauss_ss5.0']

        if show_only_p90:
            subset = {k: v for k, v in results_spatial_temporal_combined.items() if v['spatial'] == s_name and v['percentile'] == 90}
        else:
            subset = {k: v for k, v in results_spatial_temporal_combined.items() if v['spatial'] == s_name}

        if not subset:
            continue

        n_rows = len(subset)
        fig, axes = plt.subplots(n_rows + 1, 5, figsize=(22, 4 * (n_rows + 1)))
        plt.subplots_adjust(hspace=0.15, wspace=0.05)

        # First row: original frame vs spatially smoothed frame
        first_res = list(subset.values())[0]
        axes[0, 0].imshow(grayscale_images[test_frame_idx], cmap='gray')
        axes[0, 0].set_title('Original Frame')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(first_res['smoothed_frames'][first_res['test_frame_idx']], cmap='gray')
        axes[0, 1].set_title(f'After {s_name}')
        axes[0, 1].axis('off')
        for c in range(2, 5):
            axes[0, c].axis('off')

        # One row per temporal filter (or per filter+percentile if showing all)
        for idx, (combo_name, res) in enumerate(subset.items(), start=1):
            f_start, f_end = res['frame_range']
            axes[idx, 0].imshow(first_res['smoothed_frames'][f_start], cmap='gray')
            axes[idx, 0].set_title(f'Smoothed Frame {f_start}')
            axes[idx, 0].axis('off')
            axes[idx, 1].imshow(first_res['smoothed_frames'][f_end], cmap='gray')
            axes[idx, 1].set_title(f'Smoothed Frame {f_end}')
            axes[idx, 1].axis('off')
            axes[idx, 2].imshow(np.abs(res['derivative']), cmap='hot')
            if show_only_p90:
                axes[idx, 2].set_title(f'{res["temporal"]}\nDeriv. Magnitude')
            else:
                axes[idx, 2].set_title(f'{res["temporal"]} p={res["percentile"]}\nDeriv. Magnitude')
            axes[idx, 2].axis('off')
            axes[idx, 3].imshow(res['mask'], cmap='gray', vmin=0, vmax=1)
            if show_only_p90:
                axes[idx, 3].set_title(f'{res["temporal"]}\nMask (thr={res["threshold"]:.1f})')
            else:
                axes[idx, 3].set_title(f'{res["temporal"]} p={res["percentile"]}\nMask (thr={res["threshold"]:.1f})')
            axes[idx, 3].axis('off')
            overlay = grayscale_images[test_frame_idx].copy()
            overlay[res['mask'] == 1] = 255
            axes[idx, 4].imshow(overlay, cmap='gray')
            if show_only_p90:
                axes[idx, 4].set_title(f'{res["temporal"]}\nOverlay')
            else:
                axes[idx, 4].set_title(f'{res["temporal"]} p={res["percentile"]}\nOverlay')
            axes[idx, 4].axis('off')

        plt.savefig(f'{output_dir}/{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()


def visualize_spatial_temporal_combined_percentile_comparison(spatial_configs, temporal_configs, frames_array,
                                         grayscale_images, test_frame_idx, percentile_values, output_dir):
    """
    Each column is a different percentile.
    Row 0: binary mask, Row 1: overlay on original frame.
    """
    for s_name, s_method, s_param in spatial_configs:
        smoothed_frames = apply_spatial_smoothing(frames_array, s_method, s_param)
        smoothed_list = [smoothed_frames[i] for i in range(len(smoothed_frames))]

        for t_name, t_method, t_param in temporal_configs:
            if t_method == 'simple':
                deriv = simple_derivative_filter(smoothed_list, test_frame_idx)
            else:
                deriv = gaussian_derivative_filter(smoothed_frames, test_frame_idx, t_param)

            if deriv is None:
                continue

            n_pct = len(percentile_values)
            fig, axes = plt.subplots(2, n_pct, figsize=(5 * n_pct, 8))
            plt.subplots_adjust(hspace=0.05, wspace=0.05)

            for i, pct in enumerate(percentile_values):
                mask, thr = threshold_percentile(deriv, pct)
                mp = 100 * np.sum(mask) / mask.size

                axes[0, i].imshow(mask, cmap='gray', vmin=0, vmax=1)
                axes[0, i].set_title(f'p={pct}, thr={thr:.1f}\n{mp:.1f}% motion', fontsize=10)
                axes[0, i].axis('off')

                overlay = grayscale_images[test_frame_idx].copy()
                overlay[mask == 1] = 255
                axes[1, i].imshow(overlay, cmap='gray')
                axes[1, i].axis('off')

            safe_s = s_name.replace(' ', '_').replace('=', '')
            safe_t = t_name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '')
            plt.savefig(f'{output_dir}/spatial_temporal_combined_percentile_{safe_s}_{safe_t}.png', dpi=150, bbox_inches='tight')
            plt.close()