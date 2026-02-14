# Image directory
IMAGE_DIR = "path to your frames directory"
# Temporal sigma values for Gaussian derivative filter
TEMPORAL_SIGMA_VALUES = [0.5, 1.5, 5.0]

# Spatial sigma values for Gaussian smoothing
SPATIAL_SIGMA_VALUES = [0.5, 1.5, 5.0]

# Percentile values for thresholding
PERCENTILE_VALUES = [80, 85, 90, 95]

# Fixed threshold values for Threshold Analysis
FIXED_THRESHOLDS = [2, 5, 10, 15, 20, 30, 50]

# K values for noise model threshold
K_VALUES = [2.0, 3.0, 4.0, 5.0]

# Spatial filter configurations
SPATIAL_CONFIGS = [
    ('Box 3x3',      'box_3x3',  None),
    ('Box 5x5',      'box_5x5',  None),
    ('Gauss ss=0.5', 'gaussian', 0.5),
    ('Gauss ss=1.5', 'gaussian', 1.5),
    ('Gauss ss=5.0', 'gaussian', 5.0),
]

# Temporal filter configurations
TEMPORAL_CONFIGS = [
    ('Simple [-1,0,1]', 'simple',   None),
    ('DoG ts=0.5',      'gaussian', 0.5),
    ('DoG ts=1.5',      'gaussian', 1.5),
    ('DoG ts=5.0',      'gaussian', 5.0),
]

# Results directory structure
RESULTS_DIR = 'results'
RESULTS_TEMPORAL_DERIVATIVES_ONLY_DIR = 'results/temporal_derivatives_only'
RESULTS_SPATIAL_TEMPORAL_COMBINED_DIR = 'results/spatial_temporal_combined'
RESULTS_THRESHOLD_ANALYSIS_DIR = 'results/threshold_analysis'

