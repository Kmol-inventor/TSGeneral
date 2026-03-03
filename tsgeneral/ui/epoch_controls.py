"""
Epoch Control Widget - for selecting time windows with unit options.
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QDoubleSpinBox, 
    QPushButton, QComboBox
)
from PySide6.QtCore import Signal


class EpochControlWidget(QWidget):
    """
    Widget for selecting epoch (start/end) with unit options.
    
    Supports:
    - Samples (integer indices)
    - Seconds 
    - Milliseconds
    
    Signals:
        epoch_changed: Emitted when epoch range changes (start_sample, end_sample)
    """
    
    epoch_changed = Signal(int, int, str)  # start, end in samples
    
    def __init__(self, max_samples: int = 10000, sample_rate: float = 128.0, parent=None):
        super().__init__(parent)
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        
        self._setup_ui()
        self._update_ranges()
    
    def _setup_ui(self):
        """Set up the controls."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 0)
        
        # Epoch label
        layout.addWidget(QLabel("Epoch:"))
        
        # Unit selector
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["samples", "seconds", "ms"])
        self.unit_combo.setCurrentIndex(0)
        self.unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        layout.addWidget(self.unit_combo)
        
        # Start value
        layout.addWidget(QLabel("Start:"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setDecimals(2)
        self.start_spin.setMinimumWidth(90)
        self.start_spin.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.start_spin)
        
        # End value
        layout.addWidget(QLabel("End:"))
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setDecimals(2)
        self.end_spin.setMinimumWidth(90)
        self.end_spin.setSpecialValueText("Full")
        self.end_spin.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.end_spin)
        
        # Info label showing sample range
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: gray;")
        layout.addWidget(self.info_label)
        
        # Apply button
        self.apply_btn = QPushButton("Apply Epoch")
        self.apply_btn.clicked.connect(self._emit_epoch)
        layout.addWidget(self.apply_btn)
    
    def _get_unit(self) -> str:
        """Get current unit."""
        return self.unit_combo.currentText()
    
    def _samples_to_display(self, samples: int) -> float:
        """Convert samples to display unit."""
        unit = self._get_unit()
        if unit == "samples":
            return float(samples)
        elif unit == "seconds":
            return samples / self.sample_rate
        else:  # ms
            return (samples / self.sample_rate) * 1000
    
    def _display_to_samples(self, value: float) -> int:
        """Convert display unit to samples."""
        unit = self._get_unit()
        if unit == "samples":
            return int(value)
        elif unit == "seconds":
            return int(value * self.sample_rate)
        else:  # ms
            return int((value / 1000) * self.sample_rate)
    
    def _update_ranges(self):
        """Update spin box ranges based on unit."""
        unit = self._get_unit()
        
        if unit == "samples":
            self.start_spin.setDecimals(0)
            self.end_spin.setDecimals(0)
            self.start_spin.setRange(0, self.max_samples)
            self.end_spin.setRange(0, self.max_samples)
            self.start_spin.setSingleStep(10)
            self.end_spin.setSingleStep(10)
        elif unit == "seconds":
            max_sec = self.max_samples / self.sample_rate
            self.start_spin.setDecimals(2)
            self.end_spin.setDecimals(2)
            self.start_spin.setRange(0, max_sec)
            self.end_spin.setRange(0, max_sec)
            self.start_spin.setSingleStep(0.1)
            self.end_spin.setSingleStep(0.1)
        else:  # ms
            max_ms = (self.max_samples / self.sample_rate) * 1000
            self.start_spin.setDecimals(1)
            self.end_spin.setDecimals(1)
            self.start_spin.setRange(0, max_ms)
            self.end_spin.setRange(0, max_ms)
            self.start_spin.setSingleStep(10)
            self.end_spin.setSingleStep(10)
        
        self._update_info()
    
    def _on_unit_changed(self):
        """Handle unit change - convert current values."""
        self._update_ranges()
        self._update_info()
    
    def _on_value_changed(self):
        """Handle spin box value changes."""
        start = self.start_spin.value()
        end = self.end_spin.value()
        
        # Ensure start < end (unless end is 0 meaning "full")
        if end > 0 and start >= end:
            self.start_spin.setValue(end - (0.01 if self._get_unit() != "samples" else 1))
        
        self._update_info()
    
    def _update_info(self):
        """Update the info label showing sample range."""
        start_samples, end_samples, tm_type = self.get_epoch()
        if end_samples == 0:
            self.info_label.setText(f"[0 - {self.max_samples}] samples")
        else:
            self.info_label.setText(f"[{start_samples} - {end_samples}] samples")
    
    def _emit_epoch(self):
        """Emit the epoch_changed signal."""
        start, end, tm_type = self.get_epoch()
        self.epoch_changed.emit(start, end, tm_type)
    
    def get_epoch(self) -> tuple[int, int, str]:
        """
        Get current epoch range in samples.
        
        Returns:
            Tuple of (start, end) sample indices.
            end=0 means full trace.
        """
        start = self._display_to_samples(self.start_spin.value())
        end = self._display_to_samples(self.end_spin.value())
        tm_type = self._get_unit()
        
        if start==0 and end == 0:
            return (None, None, tm_type)
        
        return (start, end, tm_type)
    
    def set_max_samples(self, max_samples: int):
        """Update the maximum sample count."""
        self.max_samples = max_samples
        self._update_ranges()
    
    def set_sample_rate(self, sample_rate: float):
        """Update the sample rate."""
        self.sample_rate = sample_rate
        self._update_ranges()
    
    def reset(self):
        """Reset to full trace."""
        self.start_spin.setValue(0)
        self.end_spin.setValue(0)
        self._emit_epoch()
