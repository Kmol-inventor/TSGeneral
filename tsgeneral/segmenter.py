"""
Segmenter - Timestamp-based trial segmentation for continuous EEG recordings.

Segments continuous recordings into trials using external timestamp markers,
designed for cases where the recording device (e.g., Emotiv) doesn't embed
event markers directly in the data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path


# Default EEG channel prefixes to auto-detect
_EEG_PREFIX = "EEG."
_EEG_CHANNEL_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]


@dataclass
class SegmentedData:
    """
    Result of segmenting a continuous recording into trials.

    Attributes:
        data: 3D array (n_trials, n_channels, n_samples) - may have ragged
              last dim if trials differ in length, in which case this is a
              list of 2D arrays instead.
        channel_names: List of channel names (e.g., ["AF3", "F7", ...])
        trial_names: List of trial labels (e.g., ["Trial 0", "Trial 1", ...])
        markers: The (start, end) timestamp pairs used for segmentation
        sample_indices: The (start_idx, end_idx) sample pairs that were extracted
        sample_rate: Sampling rate in Hz
        uniform: True if all trials have the same number of samples
    """
    data: Union[np.ndarray, list[np.ndarray]]
    channel_names: list[str]
    trial_names: list[str]
    markers: list[tuple[float, float]]
    sample_indices: list[tuple[int, int]]
    sample_rate: float
    uniform: bool

    @property
    def n_trials(self) -> int:
        if isinstance(self.data, np.ndarray):
            return self.data.shape[0]
        return len(self.data)

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    def get_trial(self, trial: Union[int, str]) -> np.ndarray:
        """
        Get all channels for a single trial.

        Args:
            trial: Trial index (int) or trial name (str)

        Returns:
            Array of shape (n_channels, n_samples)
        """
        idx = self._resolve_trial(trial)
        if isinstance(self.data, np.ndarray):
            return self.data[idx]
        return self.data[idx]

    def get_channel(self, channel: Union[int, str]) -> np.ndarray:
        """
        Get a single channel across all trials.

        Args:
            channel: Channel index (int) or channel name (str)

        Returns:
            Array of shape (n_trials, n_samples) if uniform,
            otherwise list of 1D arrays
        """
        ch_idx = self._resolve_channel(channel)
        if isinstance(self.data, np.ndarray):
            return self.data[:, ch_idx, :]
        return [trial[ch_idx] for trial in self.data]

    def select(
        self,
        trials: Optional[Union[list[int], list[str], int, str]] = None,
        channels: Optional[Union[list[int], list[str], int, str]] = None,
    ) -> "SegmentedData":
        """
        Create a new SegmentedData with a subset of trials and/or channels.

        Args:
            trials: Trial indices/names to keep (None = all)
            channels: Channel indices/names to keep (None = all)

        Returns:
            New SegmentedData with the selection applied
        """
        # Resolve trial indices
        if trials is None:
            t_indices = list(range(self.n_trials))
        else:
            if not isinstance(trials, list):
                trials = [trials]
            t_indices = [self._resolve_trial(t) for t in trials]

        # Resolve channel indices
        if channels is None:
            ch_indices = list(range(self.n_channels))
        else:
            if not isinstance(channels, list):
                channels = [channels]
            ch_indices = [self._resolve_channel(c) for c in channels]

        # Slice data
        if isinstance(self.data, np.ndarray):
            new_data = self.data[np.ix_(t_indices, ch_indices)]
        else:
            new_data = [self.data[t][ch_indices] for t in t_indices]

        # Check uniformity
        if isinstance(new_data, np.ndarray):
            uniform = True
        else:
            lengths = [d.shape[-1] for d in new_data]
            uniform = len(set(lengths)) == 1
            if uniform:
                new_data = np.array(new_data)

        return SegmentedData(
            data=new_data,
            channel_names=[self.channel_names[i] for i in ch_indices],
            trial_names=[self.trial_names[i] for i in t_indices],
            markers=[self.markers[i] for i in t_indices],
            sample_indices=[self.sample_indices[i] for i in t_indices],
            sample_rate=self.sample_rate,
            uniform=uniform,
        )

    def to_inspector_data(
        self,
        mode: str = "trials",
        trial: Optional[Union[int, str]] = None,
        channel: Optional[Union[int, str]] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Prepare data for Inspector as a 2D array (rows x samples).

        Args:
            mode: How to organize rows:
                - "trials": One row per trial. Requires a single channel selected.
                - "channels": One row per channel. Requires a single trial selected.
                - "flat": All trial x channel combinations as rows.
            trial: Which trial to use (required for mode="channels")
            channel: Which channel to use (required for mode="trials")

        Returns:
            (data_2d, row_names) tuple ready for Inspector
        """
        if mode == "channels":
            if trial is None:
                raise ValueError("mode='channels' requires a trial selection")
            trial_data = self.get_trial(trial)  # (n_channels, n_samples)
            row_names = list(self.channel_names)
            return trial_data, row_names

        elif mode == "trials":
            if channel is None:
                raise ValueError("mode='trials' requires a channel selection")
            ch_data = self.get_channel(channel)  # (n_trials, n_samples) or list
            if isinstance(ch_data, list):
                # Pad to uniform length for Inspector
                max_len = max(len(t) for t in ch_data)
                padded = np.full((len(ch_data), max_len), np.nan)
                for i, t in enumerate(ch_data):
                    padded[i, :len(t)] = t
                ch_data = padded
            row_names = list(self.trial_names)
            return ch_data, row_names

        elif mode == "flat":
            rows = []
            row_names = []
            for t_idx in range(self.n_trials):
                trial_data = self.get_trial(t_idx)
                for ch_idx in range(self.n_channels):
                    if trial_data.ndim == 2:
                        rows.append(trial_data[ch_idx])
                    else:
                        rows.append(trial_data)
                    row_names.append(
                        f"{self.trial_names[t_idx]} | {self.channel_names[ch_idx]}"
                    )
            if self.uniform:
                data_2d = np.array(rows)
            else:
                max_len = max(len(r) for r in rows)
                data_2d = np.full((len(rows), max_len), np.nan)
                for i, r in enumerate(rows):
                    data_2d[i, :len(r)] = r
            return data_2d, row_names

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'trials', 'channels', or 'flat'.")

    def _resolve_trial(self, trial: Union[int, str]) -> int:
        if isinstance(trial, str):
            try:
                return self.trial_names.index(trial)
            except ValueError:
                raise KeyError(f"Trial '{trial}' not found. Available: {self.trial_names}")
        return trial

    def _resolve_channel(self, channel: Union[int, str]) -> int:
        if isinstance(channel, str):
            # Try exact match first, then case-insensitive
            if channel in self.channel_names:
                return self.channel_names.index(channel)
            lower_names = [n.lower() for n in self.channel_names]
            if channel.lower() in lower_names:
                return lower_names.index(channel.lower())
            raise KeyError(
                f"Channel '{channel}' not found. Available: {self.channel_names}"
            )
        return channel

    def summary(self) -> str:
        """Print a summary of the segmented data."""
        lines = [
            f"SegmentedData: {self.n_trials} trials, {self.n_channels} channels",
            f"  Sample rate: {self.sample_rate} Hz",
            f"  Uniform trials: {self.uniform}",
        ]
        for i in range(self.n_trials):
            trial_data = self.get_trial(i)
            n_samp = trial_data.shape[-1]
            dur = n_samp / self.sample_rate
            start, end = self.markers[i]
            lines.append(
                f"  {self.trial_names[i]}: {n_samp} samples ({dur:.2f}s) "
                f"[{start:.3f} -> {end:.3f}]"
            )
        return "\n".join(lines)


class Segmenter:
    """
    Segments a continuous EEG recording into trials using external timestamp markers.

    Designed for Emotiv CSV exports but works with any CSV that has a timestamp
    column and data columns.

    Example:
        seg = Segmenter("recording.csv")
        seg.add_markers([
            (1751733490.0, 1751733495.0),  # Trial 1: 5 seconds
            (1751733500.0, 1751733505.0),  # Trial 2: 5 seconds
        ])
        result = seg.extract()

        # View one channel across all trials
        data, names = result.to_inspector_data(mode="trials", channel="AF3")
        Inspector(data, pipeline, row_names=names).run()

        # View all channels for one trial
        data, names = result.to_inspector_data(mode="channels", trial=0)
        Inspector(data, pipeline, row_names=names, row_label="Channel").run()
    """

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        timestamp_col: str = "Timestamp",
        channel_cols: Optional[list[str]] = None,
        sample_rate: Optional[float] = None,
    ):
        """
        Initialize the Segmenter.

        Args:
            csv_path: Path to the continuous recording CSV. If None, use
                      load_data() later.
            timestamp_col: Name of the timestamp column (Unix epoch floats)
            channel_cols: List of column names to use as channels. If None,
                          auto-detects EEG columns (columns starting with "EEG.").
            sample_rate: Sampling rate in Hz. If None, auto-detected from
                         the CSV metadata header or inferred from timestamps.
        """
        self.timestamp_col = timestamp_col
        self._channel_cols = channel_cols
        self._sample_rate_override = sample_rate

        self._df: Optional[pd.DataFrame] = None
        self._timestamps: Optional[np.ndarray] = None
        self._channel_names: Optional[list[str]] = None
        self._channel_data: Optional[np.ndarray] = None
        self._sample_rate: Optional[float] = None
        self._markers: list[tuple[float, float]] = []
        self._trial_names: list[str] = []
        self._metadata: dict = {}

        if csv_path is not None:
            self.load_csv(csv_path)

    @property
    def sample_rate(self) -> float:
        if self._sample_rate is None:
            raise ValueError("No data loaded yet")
        return self._sample_rate

    @property
    def channel_names(self) -> list[str]:
        if self._channel_names is None:
            raise ValueError("No data loaded yet")
        return self._channel_names

    @property
    def duration(self) -> float:
        """Total recording duration in seconds."""
        if self._timestamps is None:
            raise ValueError("No data loaded yet")
        return float(self._timestamps[-1] - self._timestamps[0])

    @property
    def time_range(self) -> tuple[float, float]:
        """(start_timestamp, end_timestamp) of the recording."""
        if self._timestamps is None:
            raise ValueError("No data loaded yet")
        return (float(self._timestamps[0]), float(self._timestamps[-1]))

    @property
    def n_samples(self) -> int:
        if self._timestamps is None:
            raise ValueError("No data loaded yet")
        return len(self._timestamps)

    @property
    def metadata(self) -> dict:
        """Metadata parsed from the CSV header (Emotiv format)."""
        return self._metadata

    def load_csv(self, csv_path: Union[str, Path]) -> "Segmenter":
        """
        Load a continuous recording from CSV.

        Handles Emotiv CSV format (metadata on line 1, headers on line 2)
        and standard CSVs (headers on line 1).

        Args:
            csv_path: Path to the CSV file

        Returns:
            self (for method chaining)
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        # Detect Emotiv metadata header
        header_row = 0
        with open(csv_path, "r") as f:
            first_line = f.readline().strip()
            if self._is_emotiv_metadata(first_line):
                self._metadata = self._parse_emotiv_metadata(first_line)
                header_row = 1

        self._df = pd.read_csv(csv_path, header=header_row)
        self._parse_loaded_data()
        return self

    def load_data(
        self,
        timestamps: np.ndarray,
        channel_data: np.ndarray,
        channel_names: list[str],
        sample_rate: Optional[float] = None,
    ) -> "Segmenter":
        """
        Load data directly from arrays (non-CSV source).

        Args:
            timestamps: 1D array of Unix epoch timestamps
            channel_data: 2D array (n_samples, n_channels)
            channel_names: List of channel names
            sample_rate: Sampling rate in Hz (inferred from timestamps if None)

        Returns:
            self (for method chaining)
        """
        self._timestamps = np.asarray(timestamps, dtype=np.float64)
        self._channel_data = np.asarray(channel_data, dtype=np.float64)
        self._channel_names = list(channel_names)

        if sample_rate is not None:
            self._sample_rate = float(sample_rate)
        elif self._sample_rate_override is not None:
            self._sample_rate = float(self._sample_rate_override)
        else:
            self._sample_rate = self._infer_sample_rate(self._timestamps)

        return self

    def add_markers(
        self,
        markers: list[tuple[float, float]],
        names: Optional[list[str]] = None,
    ) -> "Segmenter":
        """
        Add trial markers as (start_timestamp, end_timestamp) pairs.

        Args:
            markers: List of (start, end) Unix epoch timestamp pairs.
            names: Optional trial names. If None, auto-named "Trial 0", etc.

        Returns:
            self (for method chaining)
        """
        start_idx = len(self._markers)
        self._markers.extend(markers)

        if names is not None:
            if len(names) != len(markers):
                raise ValueError(
                    f"Got {len(names)} names for {len(markers)} markers"
                )
            self._trial_names.extend(names)
        else:
            for i in range(len(markers)):
                self._trial_names.append(f"Trial {start_idx + i}")

        return self

    def clear_markers(self) -> "Segmenter":
        """Remove all markers."""
        self._markers.clear()
        self._trial_names.clear()
        return self

    def extract(
        self,
        channels: Optional[Union[list[str], list[int]]] = None,
    ) -> SegmentedData:
        """
        Extract trials from the continuous recording using the added markers.

        Args:
            channels: Subset of channels to extract (None = all).
                      Can be names (e.g., ["AF3", "F7"]) or indices.

        Returns:
            SegmentedData with the segmented trials
        """
        if self._timestamps is None:
            raise ValueError("No data loaded. Call load_csv() or load_data() first.")
        if not self._markers:
            raise ValueError("No markers added. Call add_markers() first.")

        # Resolve channel selection
        if channels is not None:
            ch_indices = []
            for ch in channels:
                if isinstance(ch, str):
                    ch_indices.append(self._resolve_channel_idx(ch))
                else:
                    ch_indices.append(ch)
            ch_names = [self._channel_names[i] for i in ch_indices]
        else:
            ch_indices = list(range(len(self._channel_names)))
            ch_names = list(self._channel_names)

        # Find sample indices for each marker pair
        trials = []
        sample_indices = []

        for start_ts, end_ts in self._markers:
            start_idx = self._find_nearest_sample(start_ts)
            end_idx = self._find_nearest_sample(end_ts)

            # Extract: (n_channels, n_samples_in_trial)
            segment = self._channel_data[start_idx:end_idx, ch_indices].T
            trials.append(segment)
            sample_indices.append((start_idx, end_idx))

        # Check if all trials are the same length
        lengths = [t.shape[1] for t in trials]
        uniform = len(set(lengths)) == 1

        if uniform:
            data = np.array(trials)  # (n_trials, n_channels, n_samples)
        else:
            data = trials  # list of (n_channels, n_samples_i) arrays

        return SegmentedData(
            data=data,
            channel_names=ch_names,
            trial_names=list(self._trial_names),
            markers=list(self._markers),
            sample_indices=sample_indices,
            sample_rate=self._sample_rate,
            uniform=uniform,
        )

    def _find_nearest_sample(self, timestamp: float) -> int:
        """Find the index of the sample closest to the given timestamp."""
        idx = np.searchsorted(self._timestamps, timestamp)
        # Check neighbors to find true nearest
        if idx == 0:
            return 0
        if idx >= len(self._timestamps):
            return len(self._timestamps) - 1
        # Compare distance to left and right neighbors
        if abs(self._timestamps[idx - 1] - timestamp) <= abs(self._timestamps[idx] - timestamp):
            return idx - 1
        return idx

    def _resolve_channel_idx(self, name: str) -> int:
        if name in self._channel_names:
            return self._channel_names.index(name)
        lower_names = [n.lower() for n in self._channel_names]
        if name.lower() in lower_names:
            return lower_names.index(name.lower())
        raise KeyError(
            f"Channel '{name}' not found. Available: {self._channel_names}"
        )

    def _parse_loaded_data(self):
        """Parse timestamps, channels, and sample rate from the loaded DataFrame."""
        df = self._df

        # Extract timestamps
        if self.timestamp_col not in df.columns:
            raise ValueError(
                f"Timestamp column '{self.timestamp_col}' not found. "
                f"Available columns: {list(df.columns[:10])}..."
            )
        self._timestamps = df[self.timestamp_col].values.astype(np.float64)

        # Detect or use specified channel columns
        if self._channel_cols is not None:
            ch_cols = self._channel_cols
        else:
            ch_cols = self._auto_detect_channels(df.columns)

        self._channel_names = [
            col.replace(_EEG_PREFIX, "") if col.startswith(_EEG_PREFIX) else col
            for col in ch_cols
        ]
        self._channel_data = df[ch_cols].values.astype(np.float64)

        # Determine sample rate
        if self._sample_rate_override is not None:
            self._sample_rate = float(self._sample_rate_override)
        elif "sampling rate" in self._metadata:
            self._sample_rate = self._metadata["sampling rate"]
        else:
            self._sample_rate = self._infer_sample_rate(self._timestamps)

    def _auto_detect_channels(self, columns: pd.Index) -> list[str]:
        """Auto-detect EEG channel columns from column names."""
        # Look for Emotiv-style "EEG.AF3" columns matching known channel names
        eeg_signal_cols = [
            c for c in columns
            if c.startswith(_EEG_PREFIX)
            and c.replace(_EEG_PREFIX, "") in _EEG_CHANNEL_NAMES
        ]
        if eeg_signal_cols:
            return eeg_signal_cols

        # Fallback: all EEG.* columns (non-Emotiv devices may use this prefix)
        eeg_cols = [c for c in columns if c.startswith(_EEG_PREFIX)]
        if eeg_cols:
            return eeg_cols

        # Look for bare channel names (AF3, F7, etc.)
        bare_matches = [c for c in columns if c in _EEG_CHANNEL_NAMES]
        if bare_matches:
            return bare_matches

        raise ValueError(
            "Could not auto-detect EEG channels. Specify channel_cols explicitly. "
            f"Available columns: {list(columns[:20])}"
        )

    @staticmethod
    def _is_emotiv_metadata(line: str) -> bool:
        """Check if the first line is an Emotiv metadata header."""
        return "headset type:" in line.lower() or "sampling rate:" in line.lower()

    @staticmethod
    def _parse_emotiv_metadata(line: str) -> dict:
        """Parse key:value pairs from the Emotiv metadata header line."""
        metadata = {}
        parts = line.split(",")
        for part in parts:
            part = part.strip()
            if ":" not in part:
                continue
            key, _, value = part.partition(":")
            key = key.strip().lower()
            value = value.strip()
            # Try to parse numeric values
            try:
                # Handle "eeg_128;mot_32" format
                if key == "sampling rate" and "eeg_" in value.lower():
                    eeg_rate = value.lower().split("eeg_")[1].split(";")[0]
                    metadata[key] = float(eeg_rate)
                elif key in ("samples", "channels"):
                    metadata[key] = int(value)
                else:
                    try:
                        metadata[key] = float(value)
                    except ValueError:
                        metadata[key] = value
            except (ValueError, IndexError):
                metadata[key] = value
        return metadata

    @staticmethod
    def _infer_sample_rate(timestamps: np.ndarray) -> float:
        """Infer sample rate from timestamp intervals."""
        if len(timestamps) < 2:
            raise ValueError("Need at least 2 samples to infer sample rate")
        median_dt = np.median(np.diff(timestamps))
        rate = 1.0 / median_dt
        # Round to common EEG rates
        common_rates = [128, 256, 512, 1024, 250, 500, 1000, 2048]
        closest = min(common_rates, key=lambda r: abs(r - rate))
        if abs(closest - rate) / rate < 0.05:  # Within 5%
            return float(closest)
        return round(rate, 1)

    def info(self) -> str:
        """Return a summary of the loaded recording."""
        if self._timestamps is None:
            return "Segmenter: no data loaded"
        lines = [
            f"Recording: {self.n_samples} samples, {self.duration:.2f}s",
            f"  Sample rate: {self.sample_rate} Hz",
            f"  Channels ({len(self.channel_names)}): {', '.join(self.channel_names)}",
            f"  Time range: {self._timestamps[0]:.3f} -> {self._timestamps[-1]:.3f}",
            f"  Markers: {len(self._markers)} trials defined",
        ]
        if self._metadata:
            lines.append(f"  Headset: {self._metadata.get('headset type', 'unknown')}")
        return "\n".join(lines)
