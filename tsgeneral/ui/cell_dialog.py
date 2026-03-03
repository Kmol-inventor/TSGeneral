"""
Cell Dialog - detailed view of a single cell's data with array browsing.
"""

import numpy as np
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QPushButton, QTabWidget, QWidget,
    QHeaderView, QGroupBox, QFormLayout, QSpinBox, QSplitter,
    QLineEdit, QScrollArea
)
from PySide6.QtCore import Qt, Signal
import pyqtgraph as pg

if TYPE_CHECKING:
    from ..inspector import Inspector


class ArrayBrowserWidget(QWidget):
    """
    Widget for browsing through array values with a mini-plot.
    
    Shows a scrollable table of values + a plot that highlights
    the current visible region.
    """
    
    def __init__(self, data: np.ndarray, sample_rate: float = 128.0, parent=None):
        super().__init__(parent)
        self.data = data
        self.sample_rate = sample_rate
        self.page_size = 100  # Number of values to show at once
        self.current_start = 0
        
        self._setup_ui()
        self._update_display()
    
    def _setup_ui(self):
        """Set up the browser UI."""
        layout = QVBoxLayout(self)
        
        # Mini plot showing full trace with selection region
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMaximumHeight(150)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('bottom', 'Sample')
        
        # Plot the full data
        time = np.arange(len(self.data))
        self.plot_widget.plot(time, self.data, pen=pg.mkPen(color=(31, 119, 180), width=1))
        
        # Add selection region
        self.region = pg.LinearRegionItem([0, self.page_size])
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self.plot_widget.addItem(self.region)
        
        layout.addWidget(self.plot_widget)
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        
        nav_layout.addWidget(QLabel("Start index:"))
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, max(0, len(self.data) - 1))
        self.start_spin.setValue(0)
        self.start_spin.valueChanged.connect(self._on_start_changed)
        nav_layout.addWidget(self.start_spin)
        
        nav_layout.addWidget(QLabel("Page size:"))
        self.page_spin = QSpinBox()
        self.page_spin.setRange(10, 1000)
        self.page_spin.setValue(self.page_size)
        self.page_spin.setSingleStep(50)
        self.page_spin.valueChanged.connect(self._on_page_size_changed)
        nav_layout.addWidget(self.page_spin)
        
        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self._go_previous)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self.next_btn)
        
        nav_layout.addStretch()
        
        # Jump to index with max info
        max_idx = len(self.data) - 1
        nav_layout.addWidget(QLabel(f"Jump to (0-{max_idx}):"))
        self.jump_edit = QLineEdit()
        self.jump_edit.setMaximumWidth(100)
        self.jump_edit.setPlaceholderText(f"0-{max_idx}")
        self.jump_edit.returnPressed.connect(self._jump_to_index)
        nav_layout.addWidget(self.jump_edit)
        
        # Jump button (in addition to Enter key)
        jump_btn = QPushButton("Go")
        jump_btn.clicked.connect(self._jump_to_index)
        jump_btn.setMaximumWidth(40)
        nav_layout.addWidget(jump_btn)
        
        layout.addLayout(nav_layout)
        
        # Values table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Index", "Time (s)", "Value", "Value (sci)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        
        layout.addWidget(self.table)
        
        # Info label
        self.info_label = QLabel()
        layout.addWidget(self.info_label)
    
    def _update_display(self):
        """Update the table and region display."""
        start = self.current_start
        end = min(start + self.page_size, len(self.data))
        
        # Update region
        self.region.blockSignals(True)
        self.region.setRegion([start, end])
        self.region.blockSignals(False)
        
        # Update table
        self.table.setRowCount(end - start)
        
        for i, idx in enumerate(range(start, end)):
            val = self.data[idx]
            time_s = idx / self.sample_rate
            
            self.table.setItem(i, 0, QTableWidgetItem(str(idx)))
            self.table.setItem(i, 1, QTableWidgetItem(f"{time_s:.4f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{val:.6f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{val:.4e}"))
        
        # Update info
        self.info_label.setText(
            f"Showing samples {start} - {end-1} of {len(self.data)} "
            f"({start/self.sample_rate:.2f}s - {(end-1)/self.sample_rate:.2f}s)"
        )
        
        # Update button states
        self.prev_btn.setEnabled(start > 0)
        self.next_btn.setEnabled(end < len(self.data))
    
    def _on_region_changed(self):
        """Handle region drag on plot."""
        region = self.region.getRegion()
        self.current_start = max(0, int(region[0]))
        self.page_size = max(10, int(region[1] - region[0]))
        
        self.start_spin.blockSignals(True)
        self.start_spin.setValue(self.current_start)
        self.start_spin.blockSignals(False)
        
        self.page_spin.blockSignals(True)
        self.page_spin.setValue(self.page_size)
        self.page_spin.blockSignals(False)
        
        self._update_display()
    
    def _on_start_changed(self, value: int):
        """Handle start index change."""
        self.current_start = value
        self._update_display()
    
    def _on_page_size_changed(self, value: int):
        """Handle page size change."""
        self.page_size = value
        self._update_display()
    
    def _go_previous(self):
        """Go to previous page."""
        self.current_start = max(0, self.current_start - self.page_size)
        self.start_spin.setValue(self.current_start)
    
    def _go_next(self):
        """Go to next page."""
        self.current_start = min(len(self.data) - self.page_size, self.current_start + self.page_size)
        self.current_start = max(0, self.current_start)
        self.start_spin.setValue(self.current_start)
    
    def _jump_to_index(self):
        """Jump to a specific index."""
        try:
            idx = int(self.jump_edit.text())
            self.current_start = max(0, min(len(self.data) - 1, idx))
            self.start_spin.setValue(self.current_start)
        except ValueError:
            pass


class CellDialog(QDialog):
    """
    Dialog for viewing detailed cell data with array browsing.
    
    Features:
    - Full statistics summary
    - Scrollable array values with mini-plot navigation
    - Copy to clipboard
    """
    
    def __init__(
        self, 
        inspector: "Inspector",
        trial: int, 
        stage: int,
        parent=None
    ):
        super().__init__(parent)
        self.inspector = inspector
        self.trial = trial
        self.stage = stage
        
        self.data = inspector.get_cell_data(trial, stage)
        self.stats = inspector.get_cell_stats(trial, stage)
        self.stage_name = inspector.pipeline[stage].name
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle(f"Array Inspector - Trial {self.trial + 1} / {self.stage_name}")
        self.setMinimumSize(700, 600)
        self.resize(800, 700)
        
        layout = QVBoxLayout(self)
        
        # Statistics group (compact)
        stats_group = QGroupBox("Statistics")
        stats_layout = QHBoxLayout(stats_group)
        stats_layout.addWidget(QLabel(f"<b>Mean:</b> {self.stats['mean']:.6f}"))
        stats_layout.addWidget(QLabel(f"<b>Std:</b> {self.stats['std']:.6f}"))
        stats_layout.addWidget(QLabel(f"<b>Min:</b> {self.stats['min']:.6f}"))
        stats_layout.addWidget(QLabel(f"<b>Max:</b> {self.stats['max']:.6f}"))
        stats_layout.addWidget(QLabel(f"<b>Samples:</b> {self.stats['length']}"))
        stats_layout.addStretch()
        layout.addWidget(stats_group)
        
        # Array browser
        self.browser = ArrayBrowserWidget(
            self.data, 
            sample_rate=self.inspector.sample_rate
        )
        layout.addWidget(self.browser)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        copy_btn = QPushButton("Copy All to Clipboard")
        copy_btn.clicked.connect(self._copy_to_clipboard)
        btn_layout.addWidget(copy_btn)
        
        copy_visible_btn = QPushButton("Copy Visible to Clipboard")
        copy_visible_btn.clicked.connect(self._copy_visible_to_clipboard)
        btn_layout.addWidget(copy_visible_btn)
        
        btn_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
    
    def _copy_to_clipboard(self):
        """Copy all array values to clipboard."""
        from PySide6.QtWidgets import QApplication
        
        text = "\n".join(f"{v:.6f}" for v in self.data)
        QApplication.clipboard().setText(text)
    
    def _copy_visible_to_clipboard(self):
        """Copy visible array values to clipboard."""
        from PySide6.QtWidgets import QApplication
        
        start = self.browser.current_start
        end = min(start + self.browser.page_size, len(self.data))
        
        text = "\n".join(f"{i}\t{self.data[i]:.6f}" for i in range(start, end))
        QApplication.clipboard().setText(text)
