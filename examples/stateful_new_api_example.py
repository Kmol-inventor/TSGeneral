"""
StatefulPipeline - New Intuitive API Example

This example demonstrates the IMPROVED StatefulPipeline API that doesn't require lambdas!
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsgeneral import Inspector, StatefulPipeline
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# Define a simple stateful filter class
# =============================================================================

class SignalProcessor:
    """
    Example stateful filter class for signal processing.
    
    This class maintains state and provides methods that modify internal data.
    """
    
    def __init__(self, data, baseline_samples=20):
        """
        Initialize the processor.
        
        Args:
            data: Input signal array
            baseline_samples: Number of samples to use for baseline
        """
        self.original = data.copy()     # Store original
        self.data = data.copy()         # Working data
        self.baseline_samples = baseline_samples
        self.normalized = None          # Will be set by normalize()
        self.binary = None              # Will be set by threshold()
    
    def subtract_baseline(self):
        """Remove baseline (mean of first N samples)."""
        baseline = self.data[:self.baseline_samples].mean()
        self.data = self.data - baseline
        print(f"  ✓ Baseline removed: {baseline:.3f}")
    
    def smooth(self, sigma=2):
        """Apply Gaussian smoothing."""
        self.data = gaussian_filter1d(self.data, sigma=sigma)
        print(f"  ✓ Smoothed with sigma={sigma}")
    
    def normalize(self):
        """Z-score normalization, stored in 'normalized' attribute."""
        mean = self.data.mean()
        std = self.data.std()
        self.normalized = (self.data - mean) / std
        print(f"  ✓ Normalized (mean={mean:.3f}, std={std:.3f})")
    
    def threshold(self, threshold_value=2.0):
        """Apply threshold, stored in 'binary' attribute."""
        self.binary = (self.normalized > threshold_value).astype(float)
        n_above = self.binary.sum()
        print(f"  ✓ Thresholded at {threshold_value} ({n_above} samples above)")


# =============================================================================
# Create pipeline using NEW INTUITIVE API (No lambda!)
# =============================================================================

print("=" * 70)
print("STATEFUL PIPELINE - NEW API (No Lambda Required!)")
print("=" * 70)

# Create pipeline
pipeline = StatefulPipeline()

# Configure the filter class - NO LAMBDA NEEDED!
print("\n1. Configuring pipeline with SignalProcessor class...")
pipeline.configure(SignalProcessor, baseline_samples=20)
print("   ✓ Configured!")

# Add stages with clear, readable syntax
print("\n2. Adding stages...")

pipeline.add_stage("Raw", read_attr="original")
print("   ✓ Stage 1: Raw (read from .original)")

pipeline.add_stage("Baseline", call_method="subtract_baseline", read_attr="data")
print("   ✓ Stage 2: Baseline (call .subtract_baseline(), read .data)")

pipeline.add_stage("Smooth", call_method="smooth", read_attr="data",
                  method_params={'sigma': 2})
print("   ✓ Stage 3: Smooth (call .smooth(sigma=2), read .data)")

pipeline.add_stage("Normalized", call_method="normalize", read_attr="normalized")
print("   ✓ Stage 4: Normalized (call .normalize(), read .normalized)")

pipeline.add_stage("Binary", call_method="threshold", read_attr="binary",
                  method_params={'threshold_value': 2.0})
print("   ✓ Stage 5: Binary (call .threshold(threshold_value=2.0), read .binary)")

print(f"\nPipeline created with {len(pipeline)} stages!")


# =============================================================================
# Generate test data and process
# =============================================================================

print("\n" + "=" * 70)
print("PROCESSING DATA")
print("=" * 70)

# Create sample data (5 trials, 100 samples each)
np.random.seed(42)
data = np.random.randn(5, 100) * 10 + 50  # Add offset and scale

print(f"\nGenerated data: {data.shape[0]} trials × {data.shape[1]} samples")
print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")

# Create Inspector
print("\n3. Creating Inspector...")
inspector = Inspector(data, pipeline)
print("   ✓ Inspector created and data processed!")


# =============================================================================
# Show results
# =============================================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print("\nStage outputs for first trial:")
for stage_name in ["Raw", "Baseline", "Smooth", "Normalized", "Binary"]:
    stage_data = inspector.get_stage_output(stage_name)[0]  # First trial
    print(f"\n{stage_name}:")
    print(f"  Shape: {stage_data.shape}")
    print(f"  Range: [{stage_data.min():.3f}, {stage_data.max():.3f}]")
    print(f"  Mean: {stage_data.mean():.3f}")


# =============================================================================
# Compare with OLD API
# =============================================================================

print("\n" + "=" * 70)
print("COMPARISON: NEW API vs OLD API")
print("=" * 70)

print("""
NEW API (Intuitive - No Lambda!):
─────────────────────────────────
pipeline = StatefulPipeline()
pipeline.configure(SignalProcessor, baseline_samples=20)
pipeline.add_stage("Raw", read_attr="original")
pipeline.add_stage("Baseline", call_method="subtract_baseline", read_attr="data")
pipeline.add_stage("Smooth", call_method="smooth", read_attr="data",
                  method_params={'sigma': 2})

✅ Clear and readable
✅ No lambda syntax needed
✅ Explicit parameter names
✅ Easy to understand for beginners


OLD API (Still works, but more complex):
────────────────────────────────────────
pipeline = StatefulPipeline(
    factory=lambda data: SignalProcessor(data, baseline_samples=20),  # ← Lambda!
    stages=[
        ("Raw", None, "original"),
        ("Baseline", "subtract_baseline", "data"),
        ("Smooth", "smooth", "data", {"sigma": 2}),
    ]
)

⚠️ Requires understanding lambda functions
⚠️ Less clear what each parameter means
⚠️ Harder for beginners
""")

print("\n" + "=" * 70)
print("READY FOR VISUALIZATION")
print("=" * 70)
print("\nThe Inspector is ready! You can now:")
print("  - View the data in TSGeneral UI")
print("  - Compare different stages")
print("  - See how data transforms through the pipeline")
print("\nRun this script and the Inspector window will open!")

# Launch Inspector UI
if __name__ == "__main__":
    inspector.run()
