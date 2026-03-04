"""
BCI Jaw Clench Detection - TSGeneral Showcase Example

This example demonstrates TSGeneral's real-world application: visualizing
a complete EEG signal processing pipeline for detecting jaw clench events
in Brain-Computer Interface (BCI) applications.

The data is from an Emotiv EEG headset (14 channels, 128 Hz sampling rate).
The pipeline processes raw EEG through several stages to detect intentional
jaw clenches that could be used as control signals.

Pipeline stages:
    1. Raw       - Original EEG signal
    2. Baseline  - DC offset removed (baseline correction)
    3. Gaussian  - High-frequency noise reduced (smoothing)
    4. Z-Score   - Amplitude normalized (standardization)
    5. Threshold - Binary activation detection
    6. Duration  - Sustained activation = "click" event

Usage:
    uv run python examples/bci_jaw_clench/run_inspector.py
"""

import os
import sys
import pandas as pd

# Add parent directory for imports when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tsgeneral import Inspector, StatefulPipeline
from bci_filters import EEGFilters


def main():
    # Load sample EEG data (14 channels × 2560 samples = 20 seconds at 128 Hz)
    data_path = os.path.join(os.path.dirname(__file__), "sample_eeg_data.csv")
    df = pd.read_csv(data_path)
    
    print(f"Loaded EEG data: {df.shape[0]} samples × {df.shape[1]} channels")
    print(f"Channels: {', '.join(df.columns)}")
    print(f"Duration: {df.shape[0] / 128:.1f} seconds at 128 Hz")
    
    # Create the StatefulPipeline for EEGFilters
    # This defines how TSGeneral will process each channel through the pipeline
    pipeline = StatefulPipeline(
        # Factory function creates a fresh EEGFilters instance for each channel
        factory=lambda data: EEGFilters(data, baseline=500, fs=128),
        stages=[
            # (Stage Name, Method to call, Attribute to read, Optional kwargs)
            ("Raw", None, "ogdata"),                              # Original data
            ("Baseline", "baseline_filt", "data"),                # DC offset removed
            ("Gaussian", "gaussian_filt", "data", {"sigma": 2, "mw": 33}),  # Smoothed
            ("Z-Score", "z_filt", "data_standard"),               # Normalized
            ("Threshold", "threshold_filt", "data_bin", {"th": 1.2}),      # Binary
            ("Duration", "duration_filt", "data_click"),          # Click detection
        ]
    )
    
    # Launch the inspector
    # - Rows = EEG channels (14 total)
    # - Columns = Processing stages (6 total)
    # - Click any cell to see the waveform
    # - Ctrl+Click to select multiple, then "Overlay" to compare
    print("\nLaunching TSGeneral Inspector...")
    print("Tips:")
    print("  - Click any cell to view the waveform")
    print("  - Ctrl+Click to select multiple cells")
    print("  - Use 'Overlay Selected' to compare channels")
    print("  - Use epoch controls to zoom into time windows")
    
    inspector = Inspector(
        data=df.values,
        pipeline=pipeline,
        trial_axis=1,              # Columns are channels (trials)
        sample_rate=128.0,         # Emotiv sampling rate
        row_label="Channel",
        row_names=list(df.columns),  # Use actual channel names (AF3, F7, etc.)
    )
    
    inspector.run()


if __name__ == "__main__":
    main()
