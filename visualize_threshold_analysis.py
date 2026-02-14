import numpy as np
import matplotlib.pyplot as plt
from thresholds import threshold_fixed, threshold_percentile, threshold_noise_model


def visualize_threshold_analysis_fixed_thresholds(test_derivatives, grayscale_images, test_frame_idx,
                                     fixed_thresholds, output_dir):
    """
    Sweep over fixed threshold values for each temporal filter.
    Each column is a different threshold. Row 0: binary mask, Row 1: overlay.
    Shows how motion detection changes with increasing threshold.
    """
    for t_name, deriv in test_derivatives.items():
        n_thr = len(fixed_thresholds)
        fig, axes = plt.subplots(2, n_thr, figsize=(4 * n_thr, 8))
        plt.subplots_adjust(hspace=0.05, wspace=0.05)

        for i, thr_val in enumerate(fixed_thresholds):
            mask, thr = threshold_fixed(deriv, thr_val)
            mp = 100 * np.sum(mask) / mask.size
            axes[0, i].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'thr={thr_val}\n{mp:.1f}%', fontsize=10)
            axes[0, i].axis('off')
            overlay = grayscale_images[test_frame_idx].copy()
            overlay[mask == 1] = 255
            axes[1, i].imshow(overlay, cmap='gray')
            axes[1, i].axis('off')

        safe = t_name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '')
        plt.savefig(f'{output_dir}/fixed_thresholds_{safe}.png', dpi=150, bbox_inches='tight')
        plt.close()


def visualize_threshold_analysis_noise_model(test_derivatives, grayscale_images, test_frame_idx,
                                 k_values, output_dir):
    """
    Visualize the noise-model adaptive threshold for each temporal filter.
    Each column is a different k multiplier. Three rows per column:
    Row 0: histogram of |derivative| with threshold line,
    Row 1: binary mask, Row 2: overlay on original frame.
    """
    for t_name, deriv in test_derivatives.items():
        abs_deriv = np.abs(deriv)
        median_val = np.median(abs_deriv)
        mad = np.median(np.abs(abs_deriv - median_val))
        sigma_noise = mad / 0.6745

        fig, axes = plt.subplots(3, len(k_values), figsize=(5 * len(k_values), 12))
        plt.subplots_adjust(hspace=0.15, wspace=0.05)

        for i, k in enumerate(k_values):
            mask, thr, _ = threshold_noise_model(deriv, k=k)
            mp = 100 * np.sum(mask) / mask.size

            # Histogram with threshold line
            axes[0, i].hist(abs_deriv.ravel(), bins=100, color='steelblue', alpha=0.7, density=True)
            axes[0, i].axvline(thr, color='red', linewidth=2, label=f'k={k}, thr={thr:.1f}')
            axes[0, i].set_title(f'k={k} (thr={thr:.2f})', fontsize=10)
            axes[0, i].set_xlim(0, np.percentile(abs_deriv, 99.5))
            axes[0, i].set_xlabel('|Pixel Derivative Value|', fontsize=9)
            axes[0, i].legend(fontsize=8)
            if i == 0:
                axes[0, i].set_ylabel('Density', fontsize=10)

            axes[1, i].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f'Motion: {mp:.1f}%', fontsize=10)
            axes[1, i].axis('off')

            overlay = grayscale_images[test_frame_idx].copy()
            overlay[mask == 1] = 255
            axes[2, i].imshow(overlay, cmap='gray')
            axes[2, i].axis('off')

        safe = t_name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '')
        plt.savefig(f'{output_dir}/Noise_model_adaptive_threshold_{safe}.png', dpi=150, bbox_inches='tight')
        plt.close()


def visualize_threshold_analysis_strategy_comparison(test_derivatives, grayscale_images, test_frame_idx,
                                       percentile_values, output_dir):
    """
    Compare all three threshold strategies side by side on the simple derivative.
    Columns: fixed thresholds, percentile thresholds, noise-model thresholds.
    Row 0: binary mask, Row 1: overlay on original frame.
    """
    deriv = test_derivatives['Simple [-1,0,1]']

    strategies = {}
    for val in [5, 10, 20]:
        m, t = threshold_fixed(deriv, val)
        strategies[f'Fixed={val}'] = (m, t)
    for pct in percentile_values:
        m, t = threshold_percentile(deriv, pct)
        strategies[f'Pctl={pct}'] = (m, t)
    for k in [2.0, 3.0, 4.0, 5.0]:
        m, t, _ = threshold_noise_model(deriv, k=k)
        strategies[f'Noise k={k}'] = (m, t)

    n_strats = len(strategies)
    fig, axes = plt.subplots(2, n_strats, figsize=(3.5 * n_strats, 7))
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    for i, (sname, (mask, thr)) in enumerate(strategies.items()):
        mp = 100 * np.sum(mask) / mask.size
        axes[0, i].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'{sname}\nthr={thr:.1f}, {mp:.1f}%', fontsize=8)
        axes[0, i].axis('off')
        overlay = grayscale_images[test_frame_idx].copy()
        overlay[mask == 1] = 255
        axes[1, i].imshow(overlay, cmap='gray')
        axes[1, i].axis('off')

    plt.savefig(f'{output_dir}/strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()