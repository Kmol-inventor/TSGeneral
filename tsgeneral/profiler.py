"""
Performance Profiler for TSGeneral pipelines.

Provides timing and memory measurement for pipeline stages to identify
bottlenecks and optimize signal processing workflows.

Example:
    from tsgeneral import Inspector, Pipeline, Profiler
    
    pipeline = Pipeline()
    pipeline.add_stage("Raw")
    pipeline.add_stage("Baseline", baseline_filter)
    pipeline.add_stage("Gaussian", gaussian_filter, sigma=2)
    
    # Profile a run
    profiler = Profiler()
    inspector = Inspector(data, pipeline, profiler=profiler)
    inspector.run()
    
    # Print performance report
    profiler.print_report()
    
    # Or get raw data
    report = profiler.get_report()
"""

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from contextlib import contextmanager


@dataclass
class StageMetrics:
    """Metrics collected for a single pipeline stage."""
    name: str
    total_time: float = 0.0
    total_memory: float = 0.0  # Peak memory in bytes
    call_count: int = 0
    
    @property
    def avg_time(self) -> float:
        """Average time per call in seconds."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    @property
    def avg_memory(self) -> float:
        """Average memory per call in bytes."""
        return self.total_memory / self.call_count if self.call_count > 0 else 0.0


@dataclass
class ProfileReport:
    """Complete profiling report for a pipeline run."""
    stages: list[StageMetrics] = field(default_factory=list)
    total_time: float = 0.0
    total_memory: float = 0.0
    overhead_time: float = 0.0  # Time not spent in stages
    
    def get_bottleneck(self) -> Optional[StageMetrics]:
        """Return the stage that takes the most time."""
        if not self.stages:
            return None
        return max(self.stages, key=lambda s: s.total_time)
    
    def get_time_percentages(self) -> dict[str, float]:
        """Return time percentage for each stage."""
        if self.total_time == 0:
            return {}
        return {
            s.name: (s.total_time / self.total_time) * 100 
            for s in self.stages
        }


class Profiler:
    """
    Performance profiler for TSGeneral pipelines.
    
    Measures execution time and memory usage for each pipeline stage,
    helping identify bottlenecks and optimization opportunities.
    
    Usage:
        profiler = Profiler()
        
        # Option 1: Use with Inspector
        inspector = Inspector(data, pipeline, profiler=profiler)
        inspector.run()
        profiler.print_report()
        
        # Option 2: Manual profiling
        profiler.start()
        for trial in trials:
            with profiler.stage("Baseline"):
                result = baseline_filter(trial)
            with profiler.stage("Gaussian"):
                result = gaussian_filter(result)
        profiler.stop()
        profiler.print_report()
    """
    
    def __init__(self, track_memory: bool = True):
        """
        Initialize the profiler.
        
        Args:
            track_memory: Whether to track memory usage (slight overhead)
        """
        self.track_memory = track_memory
        self._stages: dict[str, StageMetrics] = {}
        self._stage_order: list[str] = []  # Preserve insertion order
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._is_running: bool = False
        self._total_stage_time: float = 0.0
    
    def reset(self):
        """Clear all collected metrics."""
        self._stages.clear()
        self._stage_order.clear()
        self._start_time = 0.0
        self._end_time = 0.0
        self._is_running = False
        self._total_stage_time = 0.0
    
    def start(self):
        """Start profiling session."""
        self.reset()
        self._start_time = time.perf_counter()
        self._is_running = True
        if self.track_memory:
            tracemalloc.start()
    
    def stop(self):
        """Stop profiling session."""
        self._end_time = time.perf_counter()
        self._is_running = False
        if self.track_memory:
            tracemalloc.stop()
    
    def _ensure_stage(self, name: str) -> StageMetrics:
        """Get or create metrics for a stage."""
        if name not in self._stages:
            self._stages[name] = StageMetrics(name=name)
            self._stage_order.append(name)
        return self._stages[name]
    
    @contextmanager
    def stage(self, name: str):
        """
        Context manager to profile a stage.
        
        Args:
            name: Stage name for reporting
            
        Example:
            with profiler.stage("Gaussian"):
                result = gaussian_filter(data)
        """
        metrics = self._ensure_stage(name)
        
        # Memory tracking
        if self.track_memory:
            tracemalloc.start()
        
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            
            # Update metrics
            metrics.total_time += elapsed
            metrics.call_count += 1
            self._total_stage_time += elapsed
            
            # Memory tracking
            if self.track_memory:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                metrics.total_memory = max(metrics.total_memory, peak)
    
    def record_stage(self, name: str, elapsed_time: float, memory_bytes: float = 0):
        """
        Manually record metrics for a stage.
        
        Args:
            name: Stage name
            elapsed_time: Time in seconds
            memory_bytes: Memory usage in bytes
        """
        metrics = self._ensure_stage(name)
        metrics.total_time += elapsed_time
        metrics.call_count += 1
        metrics.total_memory = max(metrics.total_memory, memory_bytes)
        self._total_stage_time += elapsed_time
    
    def get_report(self) -> ProfileReport:
        """
        Generate a profiling report.
        
        Returns:
            ProfileReport with all collected metrics
        """
        total_time = self._end_time - self._start_time if self._end_time > 0 else self._total_stage_time
        
        # Calculate total memory (max across stages)
        total_memory = max((s.total_memory for s in self._stages.values()), default=0)
        
        # Preserve stage order
        ordered_stages = [self._stages[name] for name in self._stage_order]
        
        return ProfileReport(
            stages=ordered_stages,
            total_time=total_time,
            total_memory=total_memory,
            overhead_time=total_time - self._total_stage_time
        )
    
    def print_report(self, show_memory: bool = True):
        """
        Print a formatted performance report to stdout.
        
        Args:
            show_memory: Whether to include memory column
        """
        report = self.get_report()
        
        if not report.stages:
            print("No profiling data collected.")
            return
        
        print("\n" + "=" * 60)
        print("PERFORMANCE REPORT")
        print("=" * 60)
        print(f"Total time: {report.total_time:.3f}s")
        if show_memory and report.total_memory > 0:
            print(f"Peak memory: {self._format_bytes(report.total_memory)}")
        if report.overhead_time > 0:
            print(f"Overhead: {report.overhead_time:.3f}s ({report.overhead_time/report.total_time*100:.1f}%)")
        print()
        
        # Table header
        if show_memory and any(s.total_memory > 0 for s in report.stages):
            print(f"{'Stage':<20} {'Time':>10} {'%':>8} {'Memory':>12} {'Calls':>8}")
            print("-" * 60)
        else:
            print(f"{'Stage':<20} {'Time':>10} {'%':>8} {'Calls':>8}")
            print("-" * 48)
        
        # Find bottleneck
        bottleneck = report.get_bottleneck()
        percentages = report.get_time_percentages()
        
        # Table rows
        for stage in report.stages:
            pct = percentages.get(stage.name, 0)
            time_str = f"{stage.total_time:.3f}s"
            pct_str = f"{pct:.1f}%"
            calls_str = str(stage.call_count)
            
            # Mark bottleneck
            marker = " ⚠️" if stage == bottleneck and pct > 50 else ""
            
            if show_memory and stage.total_memory > 0:
                mem_str = self._format_bytes(stage.total_memory)
                print(f"{stage.name:<20} {time_str:>10} {pct_str:>8} {mem_str:>12} {calls_str:>8}{marker}")
            else:
                print(f"{stage.name:<20} {time_str:>10} {pct_str:>8} {calls_str:>8}{marker}")
        
        print()
        
        # Bottleneck warning
        if bottleneck and percentages.get(bottleneck.name, 0) > 50:
            pct = percentages[bottleneck.name]
            print(f"⚠️  Bottleneck: {bottleneck.name} ({pct:.1f}% of total time)")
        
        print("=" * 60 + "\n")
    
    def get_summary(self) -> str:
        """
        Get a one-line summary of the profiling results.
        
        Returns:
            Summary string like "5.2s total, bottleneck: Gaussian (80%)"
        """
        report = self.get_report()
        if not report.stages:
            return "No profiling data"
        
        bottleneck = report.get_bottleneck()
        percentages = report.get_time_percentages()
        
        if bottleneck:
            pct = percentages.get(bottleneck.name, 0)
            return f"{report.total_time:.2f}s total, bottleneck: {bottleneck.name} ({pct:.0f}%)"
        return f"{report.total_time:.2f}s total"
    
    @staticmethod
    def _format_bytes(bytes_val: float) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if abs(bytes_val) < 1024:
                return f"{bytes_val:.1f} {unit}"
            bytes_val /= 1024
        return f"{bytes_val:.1f} TB"


def profile_function(func: Callable, *args, **kwargs) -> tuple[Any, float, float]:
    """
    Profile a single function call.
    
    Args:
        func: Function to profile
        *args: Arguments to pass to function
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Tuple of (result, elapsed_time, peak_memory_bytes)
    
    Example:
        result, elapsed, memory = profile_function(gaussian_filter, data, sigma=2)
        print(f"Took {elapsed:.3f}s, used {memory/1024/1024:.1f} MB")
    """
    tracemalloc.start()
    start = time.perf_counter()
    
    result = func(*args, **kwargs)
    
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return result, elapsed, peak
