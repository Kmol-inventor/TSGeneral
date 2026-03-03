"""
TSGeneral Basic Usage Example

This example demonstrates how to use TSGeneral to inspect
EEG data as it passes through a filter pipeline.
"""

import numpy as np
import sys
import os
import debugpy

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
debugpy.listen(5678)  # Must match port in launch.json
print("⏸️  Debugger listening on port 5678. Attach VS Code debugger now!")
#debugpy.wait_for_client()  # Uncomment to pause until debugger attaches

from tsgeneral import Inspector, Pipeline


# =============================================================================
# Define some example filter functions
# =============================================================================

def baseline_filter(data: np.ndarray, baseline_samples: int = 128) -> np.ndarray:
    """Remove baseline (mean of first N samples)."""
    baseline = np.mean(data[:baseline_samples])
    return data - baseline


def gaussian_filter(data: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply Gaussian smoothing."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(data, sigma=sigma)


def zscore_filter(data: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    return (data - np.mean(data)) / np.std(data)


def threshold_filter(data: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    """Binary threshold filter."""
    return (data > threshold).astype(float)


# =============================================================================
# Generate some fake EEG-like data
# =============================================================================

def generate_fake_eeg(n_trials: int = 5, n_samples: int = 1000, fs: int = 128) -> np.ndarray:
    """
    Generate fake EEG-like data with some structure.
    
    Returns:
        Array of shape (n_trials, n_samples)
    """
    np.random.seed(42)
    
    data = np.zeros((n_trials, n_samples))
    
    for trial in range(n_trials):
        # Base noise
        noise = np.random.randn(n_samples) * 0.5
        
        # Add some oscillation (simulating alpha rhythm ~10Hz)
        t = np.arange(n_samples) / fs
        alpha = 0.8 * np.sin(2 * np.pi * 10 * t)
        
        # Add baseline offset (different per trial)
        offset = np.random.uniform(-0.5, 0.5)
        
        # Add event-related potential (ERP) around sample 400
        erp = np.zeros(n_samples)
        erp_start = 350 + np.random.randint(-20, 20)
        erp_peak = erp_start + 50
        erp[erp_start:erp_peak] = np.linspace(0, 2, erp_peak - erp_start)
        erp[erp_peak:erp_peak + 100] = 2 * np.exp(-np.arange(100) / 30)
        
        data[trial] = noise + alpha + offset + erp
    
    return data


# =============================================================================
# Main example
# =============================================================================

if __name__ == "__main__":
    # Create the pipeline
    pipeline = Pipeline()
    pipeline.add_stage("Raw")  # No filter, just raw data
    pipeline.add_stage("Baseline", baseline_filter, {"baseline_samples":128})
    pipeline.add_stage("Gaussian", gaussian_filter, sigma=2.0)
    pipeline.add_stage("Z-Score", zscore_filter)
    pipeline.add_stage("Threshold", threshold_filter, threshold=1.5)
    
    # Generate fake data
    print("Generating fake EEG data...")
    data = generate_fake_eeg(n_trials=10, n_samples=1000, fs=128)
    print(f"Data shape: {data.shape}")
    
    # Create inspector and launch
    print("Launching TSGeneral Inspector...")
    inspector = Inspector(
        data=data,
        pipeline=pipeline,
        sample_rate=128.0
    )
    
    inspector.run()
