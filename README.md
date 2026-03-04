# TSGeneral

**Time-Series General** - A lightweight native desktop app for inspecting time-series data through filter pipelines.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)

## Overview

TSGeneral is a debugging and inspection tool for time-series data. It allows you to:

- **Visualize data transformations** - See how your data changes at each filter stage
- **Compare trials/channels** - View multiple signals side-by-side or overlaid
- **Inspect values** - Click any cell to see the waveform and statistics
- **Define epochs** - Focus on specific time windows across all stages

Think of it as "DB Browser for time-series data" - a purpose-built tool for understanding signal processing pipelines.

---

## Real-World Example: BCI Jaw Clench Detection

TSGeneral was built for practical signal processing work. Here's a complete example using **real EEG data** from an Emotiv headset to detect jaw clenches for Brain-Computer Interface control:

```python
from tsgeneral import Inspector, StatefulPipeline
from examples.bci_jaw_clench.bci_filters import EEGFilters
import pandas as pd

# Load 14-channel EEG data (128 Hz sampling rate)
df = pd.read_csv("examples/bci_jaw_clench/sample_eeg_data.csv")

# Define the signal processing pipeline
pipeline = StatefulPipeline(
    factory=lambda data: EEGFilters(data, baseline=500, fs=128),
    stages=[
        ("Raw", None, "ogdata"),                                    # Original signal
        ("Baseline", "baseline_filt", "data"),                      # DC offset removed
        ("Gaussian", "gaussian_filt", "data", {"sigma": 2, "mw": 33}),  # Smoothed
        ("Z-Score", "z_filt", "data_standard"),                     # Normalized
        ("Threshold", "threshold_filt", "data_bin", {"th": 1.2}),   # Binary activation
        ("Duration", "duration_filt", "data_click"),                # Click detection
    ]
)

# Launch the inspector - rows are channels, columns are pipeline stages
inspector = Inspector(
    data=df.values,
    pipeline=pipeline,
    trial_axis=1,
    sample_rate=128.0,
    row_label="Channel",
    row_names=list(df.columns),
)
inspector.run()
```

**Run the included example:**
```bash
uv run python examples/bci_jaw_clench/run_inspector.py
```

The pipeline processes raw EEG through baseline correction, Gaussian smoothing, Z-score normalization, thresholding, and duration filtering to detect intentional jaw clenches that could serve as control signals.

---

## Installation

```bash
# From source (recommended)
git clone https://github.com/Kmol-inventor/tsgeneral.git
cd tsgeneral
uv sync

# Or using pip
pip install tsgeneral
```

## Quick Start (Simple Pipeline)

For simple stateless filter functions, use the basic `Pipeline`:

```python
from tsgeneral import Inspector, Pipeline
import numpy as np

def baseline_filter(data, baseline_samples=128):
    return data - np.mean(data[:baseline_samples])

def smooth_filter(data, sigma=2.0):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(data, sigma=sigma)

pipeline = Pipeline()
pipeline.add_stage("Raw")
pipeline.add_stage("Baseline", baseline_filter, baseline_samples=128)
pipeline.add_stage("Smoothed", smooth_filter, sigma=2.0)

# Data shape: (n_trials, n_samples)
data = np.random.randn(10, 1000)
Inspector(data, pipeline, sample_rate=128.0).run()
```

## Features

- **Pipeline Grid** - Rows are trials/channels, columns are processing stages
- **Click to Plot** - Click any cell to view the waveform
- **Multi-Selection** - Ctrl+Click to select multiple, then overlay or average
- **Epoch Controls** - Zoom into specific time windows
- **Statistics** - Each cell shows μ, σ, min, max

## Filter Signatures

**Stateless filters** (simple functions):
```python
def my_filter(data: np.ndarray, **params) -> np.ndarray:
    return processed_data
```

**Stateful filters** (class-based, like `EEGFilters`):
```python
# Use StatefulPipeline with a factory function
pipeline = StatefulPipeline(
    factory=lambda data: MyFilterClass(data),
    stages=[("StageName", "method_name", "output_attribute", {params})]
)
```

## Project Structure

```
tsgeneral/
├── tsgeneral/          # Core library
│   ├── inspector.py    # Main Inspector class
│   ├── pipeline.py     # Stateless Pipeline
│   ├── stateful_pipeline.py  # StatefulPipeline for class-based filters
│   └── ui/             # PySide6 GUI components
└── examples/
    ├── basic_usage.py  # Simple getting started example
    └── bci_jaw_clench/ # Real EEG processing example
        ├── bci_filters.py      # Custom EEG filter pipeline
        ├── sample_eeg_data.csv # 14-channel Emotiv EEG data
        └── run_inspector.py    # Launch the demo
```

## Dependencies

- **numpy** / **pandas** - Data handling
- **PySide6** - Native Qt GUI
- **pyqtgraph** - Fast interactive plotting
- **scipy** - Signal processing

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Developed by [Kmol-inventor](https://github.com/Kmol-inventor) with AI collaboration (Claude) for Qt GUI development.
