"""
Pipeline vs StatefulPipeline - Clear Comparison

This example shows the difference between the two pipeline types.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsgeneral import Inspector, Pipeline, StatefulPipeline
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# Example 1: Pipeline (Function-based)
# =============================================================================
print("=" * 70)
print("PIPELINE (Function-based)")
print("=" * 70)

# Create sample data
data = np.random.randn(5, 100)  # 5 trials, 100 samples each

# Define simple filter functions
def baseline_subtract(x):
    """Subtract mean of first 20 samples."""
    baseline = x[:20].mean()
    return x - baseline

def zscore(x):
    """Z-score normalization."""
    return (x - x.mean()) / x.std()

# Create Pipeline with functions
pipeline = Pipeline()
pipeline.add_stage("Raw")  # No function = passthrough
pipeline.add_stage("Baseline", baseline_subtract)
pipeline.add_stage("Gaussian", gaussian_filter1d, sigma=2)
pipeline.add_stage("Z-score", zscore)

print("\nPipeline stages:")
for i, stage in enumerate(pipeline.stages, 1):
    print(f"  {i}. {stage.name}")
    if stage.func:
        print(f"     Function: {stage.func.__name__}")
        if stage.params:
            print(f"     Params: {stage.params}")

# Use with Inspector
inspector = Inspector(data, pipeline)
print(f"\nProcessed {data.shape[0]} trials through {len(pipeline)} stages")


# =============================================================================
# Example 2: StatefulPipeline (Class-based)
# =============================================================================
print("\n" + "=" * 70)
print("STATEFUL PIPELINE (Class-based)")
print("=" * 70)

# Example stateful filter class
class SignalProcessor:
    """
    Example of a stateful filter class.
    
    This class maintains internal state and has methods that modify
    different attributes.
    """
    
    def __init__(self, data, baseline_samples=20):
        self.original = data.copy()  # Store original
        self.data = data.copy()      # Working data
        self.baseline_samples = baseline_samples
        self.normalized = None       # Will be set by methods
    
    def subtract_baseline(self):
        """Subtract baseline from data."""
        baseline = self.data[:self.baseline_samples].mean()
        self.data = self.data - baseline
    
    def smooth(self, sigma=2):
        """Gaussian smoothing."""
        self.data = gaussian_filter1d(self.data, sigma=sigma)
    
    def normalize(self):
        """Z-score normalization, stored in 'normalized' attribute."""
        self.normalized = (self.data - self.data.mean()) / self.data.std()


# Create StatefulPipeline with factory
stateful_pipeline = StatefulPipeline()

# NEW API - Configure the filter class (NO LAMBDA!)
stateful_pipeline.configure(SignalProcessor, baseline_samples=20)

# NEW API - Add stages with clear syntax
stateful_pipeline.add_stage("Raw", read_attr="original")
stateful_pipeline.add_stage("Baseline", call_method="subtract_baseline", read_attr="data")
stateful_pipeline.add_stage("Smooth", call_method="smooth", read_attr="data",
                           method_params={'sigma': 2})
stateful_pipeline.add_stage("Normalized", call_method="normalize", read_attr="normalized")

print("\nStatefulPipeline stages:")
for i, stage in enumerate(stateful_pipeline.stages, 1):
    print(f"  {i}. {stage.name}")
    print(f"     Method: {stage.method_name or '(none - just read attribute)'}")
    print(f"     Read from: .{stage.output_attr}")
    if stage.params:
        print(f"     Params: {stage.params}")

# Use with Inspector
inspector2 = Inspector(data, stateful_pipeline)
print(f"\nProcessed {data.shape[0]} trials through {len(stateful_pipeline)} stages")


# =============================================================================
# Key Takeaways
# =============================================================================
print("\n" + "=" * 70)
print("KEY DIFFERENCES")
print("=" * 70)

print("""
┌─────────────────────┬──────────────────────┬───────────────────────┐
│ Aspect              │ Pipeline             │ StatefulPipeline      │
├─────────────────────┼──────────────────────┼───────────────────────┤
│ Filter Type         │ Pure functions       │ Class with methods    │
│ State Management    │ No state (stateless) │ Maintains state       │
│ Data Flow           │ output = f(input)    │ instance.method()     │
│ Configuration       │ Just add functions   │ Use .configure()      │
│ Use Case            │ Simple filters       │ Complex filter classes│
└─────────────────────┴──────────────────────┴───────────────────────┘

PIPELINE - When to use:
  ✓ You have standalone filter functions
  ✓ Each filter is independent (no state between calls)
  ✓ Simple, clean, functional approach
  ✓ Example: scipy.ndimage.gaussian_filter1d

STATEFUL PIPELINE - When to use:
  ✓ You have a filter CLASS (like EEGfilters)
  ✓ The class maintains state across method calls
  ✓ Methods modify internal attributes
  ✓ You need to read different attributes at each stage
  ✓ Example: EEGfilters class with .baseline_filt(), .gaussian_filt()

NEW API (Recommended):
  pipeline = StatefulPipeline()
  pipeline.configure(MyFilterClass, param1=value1)  # ← No lambda!
  pipeline.add_stage("Stage1", call_method="method1", read_attr="data")
  
  ✅ Clear and intuitive
  ✅ No lambda syntax needed
  ✅ Easy to understand

OLD API (Still works):
  pipeline = StatefulPipeline(
      factory=lambda data: MyFilterClass(data, param1=value1)  # ← Lambda
  )
  
  ⚠️ More complex
  ⚠️ Requires lambda knowledge
""")

print("\n" + "=" * 70)
print("Ready! Both inspectors are available for visualization.")
print("=" * 70)
