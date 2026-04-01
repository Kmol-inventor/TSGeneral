"""
Segmenter Example - Timestamp-based trial segmentation

Demonstrates how to segment a continuous EEG recording into trials
using external timestamp markers (e.g., from your experiment code).

This solves the common problem where Emotiv recordings don't have
event markers (no subscription tier) but your experiment separately
logged when each trial started and ended.

Usage:
    uv run python examples/segmenter_example.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsgeneral import Segmenter, Inspector, StatefulPipeline


# =============================================================================
# 1. Load a continuous Emotiv recording
# =============================================================================

# Point this to your actual Emotiv CSV export
CSV_PATH = r"C:\Users\theko\Javra-bci\recordings\Jaw_clench_test_5_nobl_EPOCX_399260_2025.07.05T18.38.05+02.00.md.csv"

seg = Segmenter(CSV_PATH)
print(seg.info())
print()

# =============================================================================
# 2. Define trial markers from your experiment
# =============================================================================

# These would come from your experiment code - (start_timestamp, end_timestamp)
# Here we'll carve up the recording into example trials
start_ts, end_ts = seg.time_range

# Example: create 3 trials of ~5 seconds each from the recording
trial_duration = 5.0
markers = []
t = start_ts + 2.0  # skip first 2 seconds
for i in range(3):
    markers.append((t, t + trial_duration))
    t += trial_duration + 2.0  # 2 second gap between trials

seg.add_markers(markers, names=["Clench 1", "Rest", "Clench 2"])
print(f"Added {len(markers)} trial markers")

# =============================================================================
# 3. Extract segmented trials
# =============================================================================

result = seg.extract()
print(result.summary())
print()

# =============================================================================
# 4. View options - pick one!
# =============================================================================

# --- Option A: One channel across all trials (compare trials) ---
# Great for: "Did the subject's AF3 response change across trials?"
data, names = result.to_inspector_data(mode="trials", channel="AF3")
print(f"Mode 'trials' (channel=AF3): {data.shape} -> {names}")

inspector = Inspector(
    data=data,
    sample_rate=seg.sample_rate,
    row_label="Trial",
    row_names=names,
)
inspector.run()

# --- Option B: All channels for one trial (compare channels) ---
# Great for: "What does the spatial distribution look like for Trial 0?"
# data, names = result.to_inspector_data(mode="channels", trial="Clench 1")
# inspector = Inspector(data=data, sample_rate=seg.sample_rate,
#                        row_label="Channel", row_names=names)
# inspector.run()

# --- Option C: Subset selection ---
# result_sub = result.select(
#     trials=["Clench 1", "Clench 2"],
#     channels=["AF3", "AF4", "F3", "F4"],
# )
# data, names = result_sub.to_inspector_data(mode="flat")
# inspector = Inspector(data=data, sample_rate=seg.sample_rate,
#                        row_label="Trial|Channel", row_names=names)
# inspector.run()

# --- Option D: With a processing pipeline ---
# from examples.bci_jaw_clench.bci_filters import EEGFilters
#
# pipeline = StatefulPipeline(
#     factory=lambda data: EEGFilters(data, baseline=64, fs=128),
#     stages=[
#         ("Raw", None, "ogdata"),
#         ("Baseline", "baseline_filt", "data"),
#         ("Gaussian", "gaussian_filt", "data", {"sigma": 2, "mw": 33}),
#         ("Z-Score", "z_filt", "data_standard"),
#     ]
# )
# data, names = result.to_inspector_data(mode="trials", channel="AF3")
# inspector = Inspector(data=data, pipeline=pipeline, sample_rate=seg.sample_rate,
#                        row_label="Trial", row_names=names)
# inspector.run()
