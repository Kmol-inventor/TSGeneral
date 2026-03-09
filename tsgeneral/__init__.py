"""
TSGeneral - Time-Series General
A lightweight native desktop app for inspecting time-series data through filter pipelines.
"""

__version__ = "0.1.0"

from .pipeline import Pipeline, Stage
from .stateful_pipeline import StatefulPipeline, StatefulStage
from .profiler import Profiler, ProfileReport, StageMetrics, profile_function
from .inspector import Inspector

__all__ = [
    "Pipeline", 
    "Stage", 
    "StatefulPipeline", 
    "StatefulStage", 
    "Profiler",
    "ProfileReport",
    "StageMetrics",
    "profile_function",
    "Inspector",
]
