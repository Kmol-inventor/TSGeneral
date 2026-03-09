"""
Stateful Pipeline module - for class-based filter pipelines.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Any, Type, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .profiler import Profiler


@dataclass
class StatefulStage:
    """
    Represents a single stage in a stateful pipeline.
    
    Attributes:
        name: Display name for the column header
        method_name: Name of the method to call on the instance (None for raw data)
        output_attr: Attribute name to read the output from after calling the method
        params: Parameters to pass to the method
    """
    name: str
    method_name: Optional[str] = None
    output_attr: str = "data"
    params: dict = field(default_factory=dict)


class StatefulPipeline:
    """
    A pipeline that works with stateful class-based filter objects.
    
    RECOMMENDED USAGE (New Intuitive API):
    
        pipeline = StatefulPipeline()
        
        # Configure the filter class and its parameters
        pipeline.configure(EEGfilters, baseline=500, fs=128)
        
        # Add stages with clear, readable syntax
        pipeline.add_stage("Raw", read_attr="ogdata")
        pipeline.add_stage("Baseline", call_method="baseline_filt", read_attr="data")
        pipeline.add_stage("Gaussian", call_method="gaussian_filt", read_attr="data",
                          method_params={'sigma': 2, 'mw': 33})
        pipeline.add_stage("Z-Score", call_method="z_filt", read_attr="data_standard")
    
    WHEN TO USE:
    - You have a FILTER CLASS (not just functions)
    - The class maintains internal state
    - Methods modify instance attributes
    - Different stages read from different attributes
    
    HOW IT WORKS:
    1. You configure the filter class using .configure()
    2. For each trial, a NEW instance is created
    3. Each stage either:
       a) Calls a method on the instance (modifies state)
       b) Just reads an attribute (no method call)
    4. Results are collected from the specified attributes
    
    DIFFERENCE FROM Pipeline:
    - Pipeline: Works with pure functions (stateless)
      pipeline.add_stage("Smooth", gaussian_filter, sigma=2)
    
    - StatefulPipeline: Works with classes (stateful)
      pipeline.configure(MyFilterClass, param=value)
      pipeline.add_stage("Step1", call_method="process", read_attr="result")
    
    Example with Custom Class:
        class MyFilters:
            def __init__(self, data, threshold=0.5):
                self.original = data.copy()
                self.data = data.copy()
                self.threshold = threshold
                self.result = None
            
            def baseline_correct(self):
                self.data = self.data - self.data[:20].mean()
            
            def smooth(self, sigma=2):
                self.data = gaussian_filter1d(self.data, sigma=sigma)
            
            def threshold_apply(self):
                self.result = self.data > self.threshold
        
        # Create pipeline with new API
        pipeline = StatefulPipeline()
        pipeline.configure(MyFilters, threshold=0.5)
        pipeline.add_stage("Raw", read_attr="original")
        pipeline.add_stage("Baseline", call_method="baseline_correct", read_attr="data")
        pipeline.add_stage("Smooth", call_method="smooth", read_attr="data", 
                          method_params={'sigma': 2})
        pipeline.add_stage("Threshold", call_method="threshold_apply", read_attr="result")
    
    OLD API (Still Supported):
        pipeline = StatefulPipeline(
            factory=lambda data: EEGfilters(data, baseline=500, fs=128),
            stages=[
                ("Raw", None, "ogdata"),
                ("Baseline", "baseline_filt", "data"),
                ("Gaussian", "gaussian_filt", "data", {"sigma": 2, "mw": 33}),
            ]
        )
    """
    
    def __init__(
        self,
        factory: Optional[Callable[[np.ndarray], Any]] = None,
        stages: Optional[list] = None
    ):
        """
        Initialize the StatefulPipeline.
        
        TWO WAYS TO USE:
        
        1. NEW API (Recommended - More Intuitive):
           pipeline = StatefulPipeline()
           pipeline.configure(EEGfilters, baseline=500, fs=128)
           pipeline.add_stage("Raw", read_attr="ogdata")
           pipeline.add_stage("Baseline", call_method="baseline_filt", read_attr="data")
        
        2. OLD API (Backward Compatible):
           pipeline = StatefulPipeline(
               factory=lambda data: EEGfilters(data, baseline=500, fs=128),
               stages=[("Raw", None, "ogdata"), ...]
           )
        
        Args:
            factory: (Optional) A callable that creates filter instances. Only needed for old API.
            stages: (Optional) List of stage tuples. Only needed for old API.
        """
        self.factory = factory
        self.filter_class = None
        self.init_params = {}
        self.stages: list[StatefulStage] = []
        
        if stages:
            for stage_def in stages:
                self.add_stage(*stage_def)
    
    def configure(
        self,
        filter_class: Type,
        **init_params
    ) -> "StatefulPipeline":
        """
        Configure the filter class and its initialization parameters.
        
        This is the RECOMMENDED way to set up StatefulPipeline - much clearer than lambda!
        
        Args:
            filter_class: The class to instantiate for each trial (e.g., EEGfilters)
            **init_params: Parameters to pass to the class __init__ (excluding data)
        
        Returns:
            self (for method chaining)
        
        Example:
            pipeline = StatefulPipeline()
            pipeline.configure(EEGfilters, baseline=500, fs=128)
            
            # This is equivalent to the old:
            # pipeline = StatefulPipeline(factory=lambda data: EEGfilters(data, baseline=500, fs=128))
        """
        self.filter_class = filter_class
        self.init_params = init_params
        
        # Create factory function from class + params
        self.factory = lambda data: filter_class(data, **init_params)
        
        return self
    
    def add_stage(
        self,
        name: str,
        method_name: Optional[str] = None,
        output_attr: Optional[str] = None,
        params: Optional[dict] = None,
        # New API parameters (more intuitive)
        call_method: Optional[str] = None,
        read_attr: Optional[str] = None,
        method_params: Optional[dict] = None
    ) -> "StatefulPipeline":
        """
        Add a stage to the pipeline.
        
        TWO WAYS TO USE:
        
        1. NEW API (Recommended - More Intuitive):
           .add_stage("Raw", read_attr="ogdata")
           .add_stage("Baseline", call_method="baseline_filt", read_attr="data")
           .add_stage("Gaussian", call_method="gaussian_filt", read_attr="data", 
                      method_params={'sigma': 2})
        
        2. OLD API (Backward Compatible):
           .add_stage("Raw", None, "ogdata")
           .add_stage("Baseline", "baseline_filt", "data")
           .add_stage("Gaussian", "gaussian_filt", "data", {"sigma": 2})
        
        Args:
            name: Display name for this stage
            
            NEW API (keyword arguments):
              call_method: Method name to call on the instance (None = don't call anything)
              read_attr: Attribute name to read the result from (default: "data")
              method_params: Parameters to pass to the method
            
            OLD API (positional arguments):
              method_name: Method to call on the instance (None to just read attribute)
              output_attr: Attribute to read the result from
              params: Parameters to pass to the method
            
        Returns:
            self (for method chaining)
        
        Examples:
            # Just read an attribute (no method call)
            pipeline.add_stage("Raw", read_attr="ogdata")
            
            # Call a method and read result
            pipeline.add_stage("Baseline", call_method="baseline_filt", read_attr="data")
            
            # Call a method with parameters
            pipeline.add_stage("Gaussian", 
                             call_method="gaussian_filt", 
                             read_attr="data",
                             method_params={'sigma': 2, 'mw': 33})
            
            # Old API still works:
            pipeline.add_stage("Baseline", "baseline_filt", "data")
        """
        # Support new intuitive API
        if call_method is not None or read_attr is not None:
            method_name = call_method
            output_attr = read_attr or "data"  # Default to "data" if not specified
            params = method_params
        
        # Support old API
        if output_attr is None:
            output_attr = "data"
        
        stage = StatefulStage(
            name=name,
            method_name=method_name,
            output_attr=output_attr,
            params=params or {}
        )
        self.stages.append(stage)
        return self
    
    def process(self, data: np.ndarray, profiler: Optional["Profiler"] = None) -> list[np.ndarray]:
        """
        Run data through all stages sequentially.
        
        Creates an instance using the factory, then calls each method
        in order and captures the output attribute.
        
        Args:
            data: Input array (single trial)
            profiler: Optional Profiler instance to collect timing data
            
        Returns:
            List of arrays, one per stage
        """
        # Create the filter instance
        instance = self.factory(data)
        
        results = []
        
        for stage in self.stages:
            if profiler is not None:
                with profiler.stage(stage.name):
                    # Call the method if specified
                    if stage.method_name is not None:
                        method = getattr(instance, stage.method_name)
                        if stage.params:
                            method(**stage.params)
                        else:
                            method()
                    
                    # Read the output attribute
                    output = getattr(instance, stage.output_attr)
                    results.append(output.copy() if hasattr(output, 'copy') else output)
            else:
                # Call the method if specified
                if stage.method_name is not None:
                    method = getattr(instance, stage.method_name)
                    if stage.params:
                        method(**stage.params)
                    else:
                        method()
                
                # Read the output attribute
                output = getattr(instance, stage.output_attr)
                results.append(output.copy() if hasattr(output, 'copy') else output)
        
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
        
        results = []
        
        for trial_idx in range(trials.shape[0]):
            trial_data = trials[trial_idx]
            stage_results = self.process(trial_data, profiler=profiler)
            results.append(stage_results)
        
        return results
    
    def __len__(self) -> int:
        return len(self.stages)
    
    def __iter__(self):
        return iter(self.stages)
    
    def __getitem__(self, idx: int) -> StatefulStage:
        return self.stages[idx]
