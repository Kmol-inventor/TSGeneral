"""
Inspector - Main public API for TSGeneral.
"""

import sys
import numpy as np
import pandas as pd
from typing import Optional, Union, TYPE_CHECKING

from .pipeline import Pipeline
from .stateful_pipeline import StatefulPipeline


# Type alias for either pipeline type
PipelineType = Union[Pipeline, StatefulPipeline]


class Inspector:
    """
    Main entry point for the TSGeneral time-series inspector.
    
    Example with stateless Pipeline:
        from tsgeneral import Inspector, Pipeline
        
        pipeline = Pipeline()
        pipeline.add_stage("Raw")
        pipeline.add_stage("Baseline", my_baseline_func)
        
        inspector = Inspector(data, pipeline)
        inspector.run()
    
    Example with StatefulPipeline (for class-based filters):
        from tsgeneral import Inspector, StatefulPipeline
        
        pipeline = StatefulPipeline(
            factory=lambda data: EEGfilters(data, baseline=500, fs=128),
            stages=[
                ("Raw", None, "ogdata"),
                ("Baseline", "baseline_filt", "data"),
                ("Gaussian", "gaussian_filt", "data", {"sigma": 2}),
            ]
        )
        
        inspector = Inspector(data, pipeline)
        inspector.run()
    """
    
    def __init__(
        self,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        pipeline: Optional[PipelineType] = None,
        trial_axis: int = 0,
        sample_rate: float = 128.0,
        row_label: str = "Trial",
        row_names: Optional[list[str]] = None,
    ):
        """
        Initialize the Inspector.
        
        Args:
            data: Input data as numpy array or pandas DataFrame.
                  Shape should be (n_rows, n_samples) if trial_axis=0
                  or (n_samples, n_rows) if trial_axis=1
            pipeline: Processing pipeline with filter stages.
                      Can be Pipeline (stateless) or StatefulPipeline (class-based).
                      If None, only raw data is shown.
            trial_axis: Which axis represents rows (0 or 1)
            sample_rate: Sampling rate in Hz (for time axis display)
            row_label: Label for rows - "Trial", "Channel", "Epoch", or custom
            row_names: Optional list of names for each row (e.g., channel names)
        """
        self.sample_rate = sample_rate
        self.trial_axis = trial_axis
        self.pipeline = pipeline or self._default_pipeline()
        self.row_label = row_label
        self.row_names = row_names
        
        # Store processed data
        self._raw_data: Optional[np.ndarray] = None
        self._processed_data: Optional[list] = None
        
        if data is not None:
            self.load_data(data)
    
    def _default_pipeline(self) -> Pipeline:
        """Create a default pipeline with just raw data."""
        pipeline = Pipeline()
        pipeline.add_stage("Raw")
        return pipeline
    
    def load_data(self, data: Union[np.ndarray, pd.DataFrame]) -> "Inspector":
        """
        Load data into the inspector.
        
        Args:
            data: Input data as numpy array or pandas DataFrame
            
        Returns:
            self (for method chaining)
        """
        # Convert DataFrame to numpy, preserve column names if available
        if isinstance(data, pd.DataFrame):
            if self.row_names is None and self.trial_axis == 1:
                # Columns become rows, use column names
                self.row_names = list(data.columns)
            data = data.values
        
        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)  # Single row
        
        # Normalize to (n_rows, n_samples)
        if self.trial_axis == 1:
            data = data.T
        
        self._raw_data = data
        self._process_data()
        
        return self
    
    def _process_data(self):
        """Process all rows through the pipeline."""
        if self._raw_data is None:
            return
        
        self._processed_data = self.pipeline.process_trials(
            self._raw_data, 
            axis=0  # Already normalized
        )
    
    @property
    def n_trials(self) -> int:
        """Number of trials in the data."""
        if self._raw_data is None:
            return 0
        return self._raw_data.shape[0]
    
    @property
    def n_samples(self) -> int:
        """Number of samples per trial."""
        if self._raw_data is None:
            return 0
        return self._raw_data.shape[1]
    
    @property
    def n_stages(self) -> int:
        """Number of stages in the pipeline."""
        return len(self.pipeline)
    
    def get_cell_data(self, trial: int, stage: int) -> np.ndarray:
        """
        Get the data for a specific cell (trial × stage).
        
        Args:
            trial: Trial index (row)
            stage: Stage index (column)
            
        Returns:
            Array of values for that cell
        """
        if self._processed_data is None:
            raise ValueError("No data loaded")
        return self._processed_data[trial][stage]
    
    def get_cell_stats(self, trial: int, stage: int) -> dict:
        """
        Get summary statistics for a cell.
        
        Args:
            trial: Trial index
            stage: Stage index
            
        Returns:
            Dict with mean, std, min, max, length
        """
        data = self.get_cell_data(trial, stage)
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "length": len(data),
        }
    
    def get_averaged_data(self, trials: list[int], stage: int) -> np.ndarray:
        """
        Get averaged data across multiple trials for a stage.
        
        Args:
            trials: List of trial indices to average
            stage: Stage index
            
        Returns:
            Averaged array
        """
        arrays = [self.get_cell_data(t, stage) for t in trials]
        # Handle potentially different lengths by using shortest
        min_len = min(len(a) for a in arrays)
        trimmed = [a[:min_len] for a in arrays]
        return np.mean(trimmed, axis=0)
    
    def run(self):
        """
        Launch the TSGeneral GUI application.
        """
        # Import here to avoid loading Qt until needed
        from .ui.main_window import launch_app
        launch_app(self)
