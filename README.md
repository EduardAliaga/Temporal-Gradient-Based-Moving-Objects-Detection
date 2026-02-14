# Motion Detection in Video Sequences

A comprehensive computer vision project for detecting motion in video sequences using temporal derivatives, spatial smoothing, and adaptive thresholding strategies. This project implements and compares various filtering and thresholding approaches for motion detection.

## Overview

This project analyzes motion in video sequences through three main components:

1. **Temporal Derivatives (No Spatial Smoothing)**: Evaluates temporal filtering methods without spatial preprocessing
2. **Spatial Smoothing + Temporal Derivatives**: Combines spatial smoothing with temporal filtering for improved motion detection
3. **Threshold Analysis & Adaptive Strategy**: Compares different thresholding strategies (fixed, percentile-based, and noise-model adaptive)

## Features

- **Temporal Filters**:
  - Simple central difference filter `[-1, 0, 1]`
  - Gaussian derivative filter (DoG) with multiple sigma values

- **Spatial Filters**:
  - Box filters (3×3, 5×5)
  - Gaussian smoothing with configurable sigma values

- **Thresholding Strategies**:
  - Fixed threshold
  - Percentile-based threshold
  - Noise-model adaptive threshold (MAD-based)

- **Metrics**:
  - Signal-to-Noise Ratio (SNR)
  - Largest Connected Component (LCC) ratio
  - Motion percentage

- **Visualizations**:
  - Comprehensive comparison plots
  - Threshold strategy comparisons
  - Percentile sensitivity analysis

## Project Structure

```
.
├── main.py                              # Main execution script
├── config.py                            # Configuration parameters
├── filters.py                           # Temporal and spatial filtering functions
├── thresholds.py                        # Thresholding strategies
├── metrics.py                           # Evaluation metrics (SNR, LCC)
├── visualize_temporal_derivatives_only.py      # Visualization for temporal derivatives
├── visualize_spatial_temporal_combined.py     # Visualization for combined filtering
├── visualize_threshold_analysis.py            # Visualization for threshold analysis
└── results/                             # Output directory (created automatically)
    ├── temporal_derivatives_only/
    ├── spatial_temporal_combined/
    └── threshold_analysis/
```

## Requirements

- Python 3.7+
- NumPy
- Pillow (PIL)
- SciPy
- Matplotlib

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install numpy pillow scipy matplotlib
```

## Configuration

Edit `config.py` to customize parameters:

- **`IMAGE_DIR`**: Path to directory containing video frames (PNG, JPG, JPEG)
- **`TEMPORAL_SIGMA_VALUES`**: Sigma values for Gaussian temporal derivative filter (default: [0.5, 1.5, 5.0])
- **`SPATIAL_SIGMA_VALUES`**: Sigma values for Gaussian spatial smoothing (default: [0.5, 1.5, 5.0])
- **`PERCENTILE_VALUES`**: Percentile thresholds to test (default: [80, 85, 90, 95])
- **`FIXED_THRESHOLDS`**: Fixed threshold values for comparison (default: [2, 5, 10, 15, 20, 30, 50])
- **`K_VALUES`**: Multiplier values for noise-model threshold (default: [2.0, 3.0, 4.0, 5.0])

## Usage

1. **Prepare your video frames**:
   - Place video frames (PNG, JPG, or JPEG format) in a directory
   - Update `IMAGE_DIR` in `config.py` to point to this directory
   - Frames should be named in alphabetical/numerical order

2. **Run the analysis**:
   ```bash
   python main.py
   ```

3. **View results**:
   - Visualizations are saved in the `results/` directory
   - Console output displays tables with metrics for each configuration
   - Results are organized by analysis type:
     - `results/temporal_derivatives_only/`: Temporal filtering results
     - `results/spatial_temporal_combined/`: Combined filtering results
     - `results/threshold_analysis/`: Threshold strategy comparisons

## Output

The script generates:

1. **Visualizations**:
   - Main comparison figures showing frames, derivatives, masks, and overlays
   - Percentile sensitivity analysis plots
   - Fixed threshold sweeps
   - Noise model adaptive threshold analysis
   - Strategy comparison plots

2. **Console Tables**:
   - **TABLE 1**: Temporal derivatives only (Filter, Percentile, Threshold, Motion%, SNR, LCC)
   - **TABLE 2**: Spatial + Temporal combinations (Spatial, Temporal, Percentile, Threshold, Motion%, SNR, LCC)
   - **THRESHOLD STRATEGY COMPARISON**: Comparison of fixed, percentile, and noise-model thresholds

## Methodology

### Temporal Derivatives (No Spatial Smoothing)

Evaluates motion detection using only temporal filtering:
- Simple `[-1, 0, 1]` central difference filter
- Gaussian derivative filters with varying temporal sigma values
- Results show the effect of temporal filtering alone

### Spatial Smoothing + Temporal Derivatives

Combines spatial preprocessing with temporal filtering:
- Spatial filters: Box (3×3, 5×5) and Gaussian smoothing
- Temporal filters: Simple and Gaussian derivative filters
- Tests all combinations to find optimal preprocessing

### Threshold Analysis & Adaptive Strategy

Compares three thresholding approaches:
- **Fixed Threshold**: Constant threshold value
- **Percentile Threshold**: Adaptive based on distribution percentiles
- **Noise Model Threshold**: Adaptive based on Median Absolute Deviation (MAD) of background noise

## Metrics

- **SNR (Signal-to-Noise Ratio)**: Ratio of foreground signal mean to background noise standard deviation
- **Motion%**: Percentage of pixels detected as motion
- **LCC (Largest Connected Component)**: Ratio of largest connected component size to total motion pixels (measures spatial coherence)

## Notes

- The script automatically selects the middle frame as the test frame
- All frames are converted to grayscale for processing
- Results directories are created automatically if they don't exist
- The project is designed to be easily extensible with additional filters or metrics

## Author

Eduard Aliaga - Computer Vision Project 1

