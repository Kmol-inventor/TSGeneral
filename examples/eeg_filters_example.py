"""
TSGeneral Example: Using StatefulPipeline with EEGfilters

This example demonstrates how to use TSGeneral with the EEGfilters class,
which is a stateful filter pipeline.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add paths for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test"))

from tsgeneral import Inspector, StatefulPipeline
from test.test_filters.bci_filters import EEGfilters


# =============================================================================
# Load real EEG data
# =============================================================================

# Load the data
data_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "test", "test_data", "Jaw_clench_5_readable.csv"
)

df = pd.read_csv(data_path)

# Select EEG channels (columns 3 onwards based on your filter_test.py)
# Columns are: [timestamp?, counter?, ...eeg_channels...]
df_eeg = df.iloc[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]

# Get channel names from columns (assuming columns 3+ are EEG channels)
eeg_columns = df_eeg.columns[3:].tolist()
print(f"EEG Channels: {eeg_columns}")

# Extract just the EEG data - shape will be (n_samples, n_channels)
eeg_data = df_eeg.iloc[:, 3:].values
print(f"EEG data shape: {eeg_data.shape} (samples × channels)")


# =============================================================================
# Option 1: View each channel through the pipeline
# =============================================================================

# Create the StatefulPipeline for EEGfilters
pipeline = StatefulPipeline(
    # Factory creates an EEGfilters instance for each channel's data
    factory=lambda data: EEGfilters(data, baseline=500, fs=128),
    stages=[
        # Raw data - just read the original data attribute
        ("Raw", None, "ogdata"),
        
        # Baseline correction
        ("Baseline", "baseline_filt", "data"),
        
        # Gaussian smoothing
        ("Gaussian", "gaussian_filt", "data", {"sigma": 2, "mw": 33}),
        
        # Z-score normalization
        ("Z-Score", "z_filt", "data_standard"),
        
        # Threshold
        ("Threshold", "threshold_filt", "data_bin", {"th": 1.2}),
        
        # Duration filter (click detection)
        ("Duration", "duration_filt", "data_click"),
    ]
)


# =============================================================================
# Launch TSGeneral Inspector
# =============================================================================

if __name__ == "__main__":
    print("Launching TSGeneral Inspector...")
    print(f"Rows will be channels, each with {eeg_data.shape[0]} samples")
    
    inspector = Inspector(
        data=eeg_data,
        pipeline=pipeline,
        trial_axis=1,  # Columns are the "rows" (channels)
        sample_rate=128.0,
        row_label="Channel",
        row_names=eeg_columns,  # Use actual channel names
    )
    
    inspector.run()
