import numpy as np
import matplotlib.pyplot as plt
from thresholds import threshold_percentile


def visualize_temporal_derivatives_only_main(results_temporal_derivatives_only, grayscale_images, test_frame_idx, output_dir):
    """
    Visualize Temporal Derivatives (No Spatial Smoothing) results using only the 90th percentile threshold.
    Each row shows one temporal filter with 5 columns: frame before, frame after, derivative magnitude, binary mask, overlay on original.
    """
    results_temporal_derivatives_only_p90 = {k: v for k, v in results_temporal_derivatives_only.items() if v['percentile'] == 90}
    n_rows = len(results_temporal_derivatives_only_p90)
    fig, axes = plt.subplots(n_rows + 1, 5, figsize=(22, 4 * (n_rows + 1)))
    plt.subplots_adjust(hspace=0.15, wspace=0.05)

    # First row: display the center frame
    axes[0, 0].imshow(grayscale_images[test_frame_idx], cmap='gray')
    axes[0, 0].set_title(f'Center Frame (t={test_frame_idx})')
    axes[0, 0].axis('off')
    for c in range(1, 5):
        axes[0, c].axis('off')

    # One row per temporal filter
    for idx, (name, res) in enumerate(results_temporal_derivatives_only_p90.items(), start=1):
        f_start, f_end = res['frame_range']
        display_name = name.split(' | p=')[0]

        axes[idx, 0].imshow(grayscale_images[f_start], cmap='gray')
        axes[idx, 0].set_title(f'Frame {f_start}')
        axes[idx, 0].axis('off')
        axes[idx, 1].imshow(grayscale_images[f_end], cmap='gray')
        axes[idx, 1].set_title(f'Frame {f_end}')
        axes[idx, 1].axis('off')
        axes[idx, 2].imshow(np.abs(res['derivative']), cmap='hot')
        axes[idx, 2].set_title(f'{display_name}\nDerivative Magnitude')
        axes[idx, 2].axis('off')
        axes[idx, 3].imshow(res['mask'], cmap='gray', vmin=0, vmax=1)
        axes[idx, 3].set_title(f'{display_name}\nMask (thr={res["threshold"]:.1f})')
        axes[idx, 3].axis('off')
        overlay = grayscale_images[test_frame_idx].copy()
        overlay[res['mask'] == 1] = 255
        axes[idx, 4].imshow(overlay, cmap='gray')
        axes[idx, 4].set_title(f'{display_name}\nOverlay')
        axes[idx, 4].axis('off')

    plt.savefig(f'{output_dir}/temporal_derivatives_only_main.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_temporal_derivatives_only_percentile_comparison(raw_temporal_derivatives_only, grayscale_images, test_frame_idx,
                                         percentile_values, output_dir):
    """
    Row 0: binary mask, Row 1: overlay on original frame.
    """
    for t_name, t_data in raw_temporal_derivatives_only.items():
        d = t_data['derivative']
        n_pct = len(percentile_values)
        fig, axes = plt.subplots(2, n_pct, figsize=(5 * n_pct, 8))
        plt.subplots_adjust(hspace=0.05, wspace=0.05)

        for i, pct in enumerate(percentile_values):
            mask, thr = threshold_percentile(d, pct)
            mp = 100 * np.sum(mask) / mask.size

            axes[0, i].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'p={pct}, thr={thr:.1f}\n{mp:.1f}% motion', fontsize=10)
            axes[0, i].axis('off')

            overlay = grayscale_images[test_frame_idx].copy()
            overlay[mask == 1] = 255
            axes[1, i].imshow(overlay, cmap='gray')
            axes[1, i].axis('off')

        safe = t_name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '')
        plt.savefig(f'{output_dir}/temporal_derivatives_only_percentile_compare_{safe}.png', dpi=150, bbox_inches='tight')
        plt.close()