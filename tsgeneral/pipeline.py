"""
Pipeline module - defines the filter pipeline structure.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Any, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .profiler import Profiler


@dataclass
class Stage:
    """
    Represents a single stage in the processing pipeline.
    
    Attributes:
        name: Display name for the column header
        func: Filter function that takes ndarray and returns ndarray.
              If None, this is a pass-through stage (e.g., "Raw" data).
        params: Optional parameters to pass to the filter function
    """
    name: str
    func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    params: dict = field(default_factory=dict)
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply this stage's filter to the input data.
        
        Args:
            data: Input array from previous stage
            
        Returns:
            Filtered array (or unchanged if func is None)
        """
        if self.func is None:
            return data.copy()
        
        if self.params:
            return self.func(data, **self.params)
        return self.func(data)


class Pipeline:
    """
    A sequence of processing stages that transform time-series data.
    
    Example:
        pipeline = Pipeline()
        pipeline.add_stage("Raw")  # Pass-through, shows original data
        pipeline.add_stage("Baseline", baseline_filter_func)
        pipeline.add_stage("Gaussian", gaussian_filter_func, sigma=2)
    """
    
    def __init__(self):
        self.stages: list[Stage] = []
    
    def add_stage(
        self, 
        name: str, 
        func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        p_dict: Union[dict, None] = None,
        **params
    ) -> "Pipeline":
        """
        Add a processing stage to the pipeline.
        
        Args:
            name: Display name for this stage (column header)
            func: Filter function. If None, data passes through unchanged.
            p_dict: Arguments in dictionary form to pass to filter function
            **params: Additional parameters to pass to the filter function
            
        Returns:
            self (for method chaining)
        """
        
        if p_dict is not None:
            stage = Stage(name=name, func=func, params=p_dict)
            
        else:
            stage = Stage(name=name, func=func, params=params)
            
        
        self.stages.append(stage)
        return self
    
    def process(self, data: np.ndarray, profiler: Optional["Profiler"] = None) -> list[np.ndarray]:
        """
        Run data through all stages sequentially.
        
        Each stage receives the output of the previous stage.
        
        Args:
            data: Input array (single trial)
            profiler: Optional Profiler instance to collect timing data
            
        Returns:
            List of arrays, one per stage
        """
        results = []
        current = data
        
        for stage in self.stages:
            if profiler is not None:
                with profiler.stage(stage.name):
                    current = stage.apply(current)
            else:
                current = stage.apply(current)
            results.append(current)
        
        return results
    
    def process_trials(
        self, 
        trials: np.ndarray, 
        axis: int = 0,
        profiler: Optional["Profiler"] = None
    ) -> list[list[np.ndarray]]:
        """
        Process multiple trials through the pipeline.
        
        Args:
            trials: 2D array of shape (n_trials, n_samples) or (n_samples, n_trials)
            axis: Which axis represents trials. 
                  0 = rows are trials (n_trials, n_samples)
                  1 = columns are trials (n_samples, n_trials)
            profiler: Optional Profiler instance to collect timing data
                  
        Returns:
            List of lists: results[trial_idx][stage_idx] = array
        """
        if axis == 1:
            trials = trials.T  # Convert to (n_trials, n_samples)
        
        n_trials = trials.shape[0]
        n_stages = len(self.stages)
        n_samples = trials.shape[1]
        
        # Pre-allocate results array
        # Note: Some filters might change array length, we'll handle that
        results = []
        
        for trial_idx in range(n_trials):
            trial_data = trials[trial_idx]
            stage_results = self.process(trial_data, profiler=profiler)
            results.append(stage_results)
        
        return results  # List of lists for now (handles variable lengths)
    
    def __len__(self) -> int:
        return len(self.stages)
    
    def __iter__(self):
        return iter(self.stages)
    
    def __getitem__(self, idx: int) -> Stage:
        return self.stages[idx]
