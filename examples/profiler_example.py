"""
Profiler Example - Measuring Pipeline Performance

This example demonstrates how to use TSGeneral's built-in profiler to
identify performance bottlenecks in your signal processing pipeline.

Usage:
    uv run python examples/profiler_example.py
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from tsgeneral import Inspector, Pipeline, Profiler, profile_function


# =============================================================================
# Define some filter functions with varying complexity
# =============================================================================

def baseline_filter(data, baseline_samples=128):
    """Fast: simple subtraction"""
    return data - np.mean(data[:baseline_samples])


def gaussian_filter(data, sigma=5.0):
    """Medium: scipy convolution"""
    return gaussian_filter1d(data, sigma=sigma)


def slow_filter(data, iterations=100):
    """Intentionally slow: multiple passes"""
    result = data.copy()
    for _ in range(iterations):
        result = np.convolve(result, [0.1, 0.8, 0.1], mode='same')
    return result


def zscore_filter(data):
    """Fast: simple statistics"""
    return (data - np.mean(data)) / np.std(data)


# =============================================================================
# Option 1: Profile using Inspector
# =============================================================================

def profile_with_inspector():
    """Profile a pipeline using Inspector's built-in profiling."""
    print("\n" + "=" * 60)
    print("OPTION 1: Profile using Inspector")
    print("=" * 60)
    
    # Create pipeline with the slow filter
    pipeline = Pipeline()
    pipeline.add_stage("Raw")
    pipeline.add_stage("Baseline", baseline_filter, baseline_samples=128)
    pipeline.add_stage("SlowFilter", slow_filter, iterations=50)  # Bottleneck!
    pipeline.add_stage("Gaussian", gaussian_filter, sigma=3.0)
    pipeline.add_stage("Z-Score", zscore_filter)
    
    # Generate test data
    data = np.random.randn(50, 2000)  # 50 trials, 2000 samples each
    
    # Create inspector with profiling enabled
    inspector = Inspector(data, pipeline, profile=True, sample_rate=128.0)
    
    # Print the performance report
    inspector.print_performance_report()
    
    # Get summary for logging
    print(f"Summary: {inspector.get_performance_summary()}")
    
    # Access raw report data
    report = inspector.get_performance_report()
    if report:
        bottleneck = report.get_bottleneck()
        print(f"\nBottleneck stage: {bottleneck.name}")
        print(f"  Total time: {bottleneck.total_time:.3f}s")
        print(f"  Average per call: {bottleneck.avg_time*1000:.2f}ms")


# =============================================================================
# Option 2: Profile manually with Profiler class
# =============================================================================

def profile_manually():
    """Use Profiler class directly for fine-grained control."""
    print("\n" + "=" * 60)
    print("OPTION 2: Manual profiling")
    print("=" * 60)
    
    profiler = Profiler(track_memory=True)
    profiler.start()
    
    # Simulate processing multiple trials
    for trial_idx in range(20):
        data = np.random.randn(1000)
        
        with profiler.stage("Baseline"):
            data = baseline_filter(data)
        
        with profiler.stage("SlowFilter"):
            data = slow_filter(data, iterations=30)
        
        with profiler.stage("Z-Score"):
            data = zscore_filter(data)
    
    profiler.stop()
    profiler.print_report()


# =============================================================================
# Option 3: Profile a single function call
# =============================================================================

def profile_single_function():
    """Profile a single function call for quick testing."""
    print("\n" + "=" * 60)
    print("OPTION 3: Profile single function")
    print("=" * 60)
    
    data = np.random.randn(10000)
    
    # Profile a single function call
    result, elapsed, memory = profile_function(
        slow_filter, data, iterations=100
    )
    
    print(f"slow_filter(iterations=100):")
    print(f"  Time: {elapsed*1000:.2f}ms")
    print(f"  Memory: {memory/1024:.1f} KB")
    print(f"  Output shape: {result.shape}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    profile_with_inspector()
    profile_manually()
    profile_single_function()
