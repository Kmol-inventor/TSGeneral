"""
BCI Signal Processing Filters for EEG Analysis

A stateful filter pipeline for processing EEG signals, particularly useful
for detecting muscle artifacts like jaw clenches in Brain-Computer Interface
applications.

Pipeline stages:
1. Baseline correction - Remove DC offset using initial samples
2. Gaussian smoothing - Reduce high-frequency noise
3. Z-score normalization - Standardize signal amplitude
4. Threshold detection - Binary classification of high-amplitude events
5. Duration filtering - Detect sustained activations (click detection)

Author: Kmol-inventor
"""

import numpy as np
import math


class EEGFilters:
    """
    Stateful EEG signal processing pipeline.
    
    Each filter method modifies internal state and returns the processed data,
    allowing inspection of intermediate results at each stage.
    
    Example:
        >>> filters = EEGFilters(eeg_trace, baseline=500, fs=128)
        >>> filters.baseline_filt()
        >>> filters.gaussian_filt(sigma=2, mw=33)
        >>> filters.z_filt()
        >>> clicks = filters.threshold_filt(th=1.2)
    """
    
    def __init__(self, data: np.ndarray, baseline: int, fs: int = 128):
        """
        Initialize the filter pipeline.
        
        Args:
            data: 1D numpy array of EEG samples
            baseline: Number of initial samples to use for baseline calculation
            fs: Sampling frequency in Hz (default: 128 for Emotiv)
        """
        self.data = data.copy()
        self.fs = fs
        self.ogdata = data.copy()  # Preserve original data
        self.baseline = baseline
        
        # Processing flags
        self.bs = 0  # Baseline applied
        self.gf = 0  # Gaussian applied
        self.zf = 0  # Z-score applied
        self.tf = 0  # Threshold applied
        self.df = 0  # Duration filter applied
        
    def baseline_filt(self) -> np.ndarray:
        """
        Apply baseline correction by subtracting the mean of initial samples.
        
        This removes DC offset and centers the signal around zero, which is
        essential for subsequent amplitude-based processing.
        
        Returns:
            Baseline-corrected signal
        """
        if not hasattr(self, 'data'):
            raise AttributeError("Baseline cannot be completed: data not found")
        
        self.base = np.mean(self.data[:self.baseline])
        self.data = np.subtract(self.data, self.base)
        self.bs = 1
        
        return self.data
    
    def gaussian_filt(self, mw: int = 7, sigma: int = 1) -> np.ndarray:
        """
        Apply Gaussian smoothing to reduce high-frequency noise.
        
        Uses a discrete Gaussian kernel convolved with the signal. The kernel
        is constructed from the Gaussian formula:
            G(x) = (1 / (σ * sqrt(2π))) * exp(- x² / (2σ²))
        
        Args:
            mw: Kernel window size (must be odd, default: 7)
            sigma: Standard deviation of Gaussian (default: 1)
            
        Returns:
            Smoothed signal
            
        Raises:
            ValueError: If mw is not odd
        """
        if not hasattr(self, 'data'):
            raise AttributeError("Gaussian cannot be completed: data not found")
        
        if mw % 2 == 0:
            raise ValueError("mw must be an odd number")
        
        tail = mw // 2
        r_tail = np.array([])
        
        # Build right side of kernel
        for i in range(tail):
            val = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(
                -(((i + 1) - 0) ** 2) / (2 * sigma ** 2)
            )
            r_tail = np.append(r_tail, val)
        
        # Mirror for left side and add center
        l_tail = np.flip(r_tail)
        center = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(0)
        kernel = np.concatenate([l_tail, np.array([center]), r_tail])
        kernel = kernel / np.sum(kernel)  # Normalize
        
        # Apply convolution (skip baseline region)
        for pos in range(len(self.data) - tail):
            if pos > self.baseline:
                window = self.data[pos - tail:pos + tail + 1]
                self.data[pos] = np.sum(window * kernel)
        
        self.gf = 1
        return self.data
    
    def z_filt(self) -> np.ndarray:
        """
        Apply Z-score normalization for consistent amplitude interpretation.
        
        Z-score formula: Z(x) = (x - μ) / σ
        
        Calculated using only post-baseline samples to avoid including
        the baseline period in the statistics.
        
        Returns:
            Z-score normalized signal (stored in self.data_standard)
        """
        mean = np.mean(self.data[self.baseline:])
        std = np.std(self.data[self.baseline:])
        self.data_standard = (self.data - mean) / std
        self.zf = 1
        
        return self.data_standard
    
    def threshold_filt(self, th: float = 1.5) -> np.ndarray:
        """
        Apply threshold to create binary activation signal.
        
        Args:
            th: Z-score threshold for activation (default: 1.5)
            
        Returns:
            Binary array (1 where signal > threshold, 0 otherwise)
        """
        self.data_bin = (self.data_standard > th).astype(int)
        self.tf = 1
        return self.data_bin
    
    def duration_filt(self, dur: int = 20, r_period_max: int = 100) -> np.ndarray:
        """
        Detect sustained activations (click detection).
        
        A "click" is registered when the threshold has been exceeded for
        at least `dur` consecutive samples. After a click, a refractory
        period prevents immediate re-triggering.
        
        Args:
            dur: Minimum duration of sustained activation (samples)
            r_period_max: Refractory period after click detection (samples)
            
        Returns:
            Binary click signal (1 at click detection moments)
        """
        self.data_click = self.data_bin * 0
        r_period = 0
        
        for i in range(len(self.data_bin)):
            if r_period > 0:
                r_period -= 1
            if np.all(self.data_bin[i - dur:i] == 1) and r_period == 0 and i > dur:
                self.data_click[i] = 1
                r_period = r_period_max
        
        self.df = 1
        return self.data_click
