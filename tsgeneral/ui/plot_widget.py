"""
Plot Widget - displays line plots using pyqtgraph with cursor tracking.
"""

import numpy as np
from typing import Optional

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Qt, Signal

import pyqtgraph as pg

import logging

# Configure logging to show in console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # This ensures console output
    ]
)


# Configure pyqtgraph for better appearance
pg.setConfigOptions(antialias=True, background='w', foreground='k')


class PlotWidget(QWidget):
    """
    Widget for displaying time-series line plots.
    
    Features:
    - Single trace or multiple overlaid traces
    - Auto-scaling
    - Interactive zoom/pan
    - Time axis in seconds (using sample rate)
    - Crosshair cursor showing X/Y values
    - Click to get exact sample/value
    """
    
    # Color palette for multiple traces
    COLORS = [
        (31, 119, 180),   # Blue
        (255, 127, 14),   # Orange
        (44, 160, 44),    # Green
        (214, 39, 40),    # Red
        (148, 103, 189),  # Purple
        (140, 86, 75),    # Brown
        (227, 119, 194),  # Pink
        (127, 127, 127),  # Gray
        (188, 189, 34),   # Olive
        (23, 190, 207),   # Cyan
    ]
    
    # Signal emitted when user clicks on plot
    point_clicked = Signal(float, float, int)  # time, value, sample_index
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._traces: list[pg.PlotDataItem] = []
        self._color_idx = 0
        self._current_data: Optional[np.ndarray] = None
        self._current_sample_rate: float = 128.0
        
        self._setup_ui()
        self._setup_crosshair()
    
    def _setup_ui(self):
        """Set up the plot widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        
        # Enable mouse interaction
        self.plot_widget.setMouseEnabled(x=True, y=True)
        
        # Add legend
        self.legend = self.plot_widget.addLegend()
        
        layout.addWidget(self.plot_widget)
        
        # Control buttons and cursor info
        btn_layout = QHBoxLayout()
        
        self.auto_range_btn = QPushButton("Auto Range")
        self.auto_range_btn.clicked.connect(self._auto_range)
        btn_layout.addWidget(self.auto_range_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear)
        btn_layout.addWidget(self.clear_btn)
        
        btn_layout.addStretch()
        
        # Cursor position label
        self.cursor_label = QLabel("Click on plot to see values")
        self.cursor_label.setStyleSheet("font-family: monospace; padding: 2px 8px; background-color: #f0f0f0; border-radius: 3px;")
        btn_layout.addWidget(self.cursor_label)
        
        layout.addLayout(btn_layout)
    
    def _setup_crosshair(self):
        """Set up crosshair cursor for tracking mouse position."""
        # Crosshair lines
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('g', width=1, style=Qt.PenStyle.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('g', width=1, style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)
        
        # Hide initially
        self.vLine.setVisible(False)
        self.hLine.setVisible(False)
        
        # Connect mouse move and click
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)
    
    def _on_mouse_moved(self, pos):
        """Handle mouse movement - update crosshair."""
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            self.vLine.setPos(mouse_point.x())
            self.hLine.setPos(mouse_point.y())
            self.vLine.setVisible(True)
            self.hLine.setVisible(True)
            
            # Update cursor label with position
            time_s = mouse_point.x()
            value = mouse_point.y()
            sample = int(time_s * self._current_sample_rate)
            
            self.cursor_label.setText(
                f"Time: {time_s:.3f}s | Sample: {sample} | Y: {value:.4f}"
            )
        else:
            self.vLine.setVisible(False)
            self.hLine.setVisible(False)
    
    def _on_mouse_clicked(self, event):
        """Handle mouse click - get exact point info."""
        pos = event.scenePos()
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            time_s = mouse_point.x()
            sample = int(time_s * self._current_sample_rate)
            
            # Get actual data value at this sample if available
            if self._current_data is not None and 0 <= sample < len(self._current_data):
                actual_value = self._current_data[sample]
                self.cursor_label.setText(
                    f"📍 Time: {time_s:.3f}s | Sample: {sample} | Value: {actual_value:.6f}"
                )
                self.cursor_label.setStyleSheet(
                    "font-family: monospace; padding: 2px 8px; background-color: #d4edda; border-radius: 3px; font-weight: bold;"
                )
                self.point_clicked.emit(time_s, actual_value, sample)
            else:
                y_val = mouse_point.y()
                self.cursor_label.setText(
                    f"📍 Time: {time_s:.3f}s | Sample: {sample} | Y: {y_val:.4f}"
                )
    
    def _get_next_color(self) -> tuple:
        """Get the next color from the palette."""
        color = self.COLORS[self._color_idx % len(self.COLORS)]
        self._color_idx += 1
        return color
    
    def _create_time_axis(self, data, sample_rate: float, start: float = None, end: float = None, tm_type: str = None) -> np.ndarray:
        """Create time axis in seconds."""
        
        if tm_type == 'seconds':
            return np.arange(start/sample_rate,end/sample_rate,((end-start)/sample_rate)/(end-start))
        else: 
            return np.arange(start, end, 1)
    
    def plot_single(
        self, 
        data: np.ndarray, 
        title: str = "",
        sample_rate: float = 128.0,
        start: float = None,
        end: float = None,
        tm_type: str = None,
        color: Optional[tuple] = None
    ):
        """
        Plot a single trace (clears existing traces).
        
        Args:
            data: 1D array of values
            title: Plot title
            sample_rate: Sampling rate for time axis
            color: RGB tuple, or None for auto-color
        """
        self.clear()
        self.sample_rate = sample_rate
        
        # Store for click lookup
       
        logging.debug(f"from plot_widget start: {start}, end: {end}, tm_type: {tm_type}")

        
        
        # set timeframe first before cutting data
        time = self._create_time_axis(data, sample_rate, start=start, end=end,tm_type=tm_type) # this time is duration of epoch, not actual time from original onset
        
        if end > 0:
            data = data[start:end]
            
        self._current_data = data.copy()
        self._current_sample_rate = sample_rate
        
        
        
        logging.debug(f"time begin = {time[0]}, time end = {time[-1]}")
        
        if color is None:
            color = self.COLORS[0]
        
        pen = pg.mkPen(color=color, width=1.5)
        trace = self.plot_widget.plot(time, data, pen=pen, name=title)
        self._traces.append(trace)
        
        self.plot_widget.setTitle(title)
        self._auto_range()
        #self.set_x_range(start,end,tm_type)
        
        
        # Reset cursor label style
        self.cursor_label.setStyleSheet(
            "font-family: monospace; padding: 2px 8px; background-color: #f0f0f0; border-radius: 3px;"
        )
    
    def add_trace(
        self, 
        data: np.ndarray, 
        name: str = "",
        sample_rate: float = 128.0,
        color: Optional[tuple] = None
    ):
        """
        Add a trace to the plot (overlays on existing).
        
        Args:
            data: 1D array of values
            name: Trace name for legend
            sample_rate: Sampling rate for time axis
            color: RGB tuple, or None for auto-color
        """
        time = self._create_time_axis(len(data), sample_rate)
        logging.debug
        
        if color is None:
            color = self._get_next_color()
        
        pen = pg.mkPen(color=color, width=1.5)
        trace = self.plot_widget.plot(time, data, pen=pen, name=name)
        self._traces.append(trace)
        
        if len(self._traces) > 1:
            self.plot_widget.setTitle(f"Overlay: {len(self._traces)} traces")
    
    def clear(self):
        """Clear all traces from the plot."""
        for trace in self._traces:
            self.plot_widget.removeItem(trace)
        self._traces.clear()
        self._color_idx = 0
        self.plot_widget.setTitle("")
        
        # Clear and recreate legend
        if self.legend is not None:
            self.legend.clear()
    
    def _auto_range(self):
        """Auto-scale the plot to fit all data."""
        self.plot_widget.autoRange()
    
    def reset_view(self):
        """Reset the view to auto-range."""
        self._auto_range()
    
    def set_x_range(self, start: float, end: float, timetype: str = None):
        """
        Set the x-axis range.
        
        Args:
            start: Start time in time-unit
            end: End time in time-unit
        """
        if timetype == 'seconds':
            self.plot_widget.setXRange((start/self.sample_rate), (end/self.sample_rate))
        else:
            self.plot_widget.setXRange(start, end)
 