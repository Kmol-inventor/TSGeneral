"""
Main window for TSGeneral application.
"""

import sys
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QToolBar, QLabel, QSpinBox, QPushButton, QStatusBar,
    QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QShortcut, QKeySequence

from .grid_widget import PipelineGridWidget
from .plot_widget import PlotWidget
from .epoch_controls import EpochControlWidget

if TYPE_CHECKING:
    from ..inspector import Inspector
    
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() 
    ]
)


class MainWindow(QMainWindow):
    """
    Main application window for TSGeneral.
    
 
    """
    
    def __init__(self, inspector: "Inspector"):
        super().__init__()
        self.inspector = inspector
        self.selected_cells: list[tuple[int, int]] = []  # (trial, stage)
        
        self._setup_window()
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_statusbar()
        
        self.resetshortcut = QShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_R), self)
        self.resetshortcut.activated.connect(self._reset_all)
        
        # Load data into grid if available
        if inspector._processed_data is not None:
            self.grid_widget.populate(inspector)
            # Auto-select and plot the first cell (Trial 1, first stage)
            self._on_cell_clicked(0, 0)
    
    def _setup_window(self):
        """Configure main window properties."""
        self.setWindowTitle("TSGeneral - Time-Series Inspector")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)
    
    def _setup_toolbar(self):
        """Create the toolbar with controls."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Epoch controls with sample rate
        self.epoch_widget = EpochControlWidget(
            max_samples=self.inspector.n_samples,
            sample_rate=self.inspector.sample_rate
        )
        self.epoch_widget.epoch_changed.connect(self._on_epoch_changed)
        toolbar.addWidget(self.epoch_widget)
        
        toolbar.addSeparator()
        
        # Average all rows button
        self.avg_all_button = QPushButton("Average All Rows")
        self.avg_all_button.clicked.connect(self._on_average_all_rows)
        self.avg_all_button.setToolTip("Average all rows (e.g., all channels) for each stage")
        toolbar.addWidget(self.avg_all_button)
        
        # Overlay selected button
        self.overlay_button = QPushButton("Overlay Selected")
        self.overlay_button.clicked.connect(self._on_overlay_selected)
        self.overlay_button.setEnabled(False)
        self.overlay_button.setToolTip("Ctrl+click cells to select multiple, then overlay them")
        toolbar.addWidget(self.overlay_button)
        
        toolbar.addSeparator()
        
        # Reset view button
        reset_button = QPushButton("Reset View")
        reset_button.clicked.connect(self._on_reset_view)
        toolbar.addWidget(reset_button)
    
    def _setup_central_widget(self):
        """Create the main layout with grid and plot."""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Splitter for resizable grid/plot areas
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Pipeline grid (top)
        self.grid_widget = PipelineGridWidget()
        self.grid_widget.cell_clicked.connect(self._on_cell_clicked)
        self.grid_widget.cell_double_clicked.connect(self._on_cell_double_clicked)
        self.grid_widget.selection_changed.connect(self._on_selection_changed)
        splitter.addWidget(self.grid_widget)
        
        # Plot area (bottom)
        self.plot_widget = PlotWidget()
        splitter.addWidget(self.plot_widget)
        
        # Set initial sizes (60% grid, 40% plot)
        splitter.setSizes([400, 300])
        
        layout.addWidget(splitter)
    
    def _setup_statusbar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._update_status("Ready - Double-click a cell to inspect array values")
    
    def _update_status(self, message: str):
        """Update status bar message."""
        self.status_bar.showMessage(message)
    
    def _on_cell_clicked(self, trial: int, stage: int):
        """Handle single cell click - show in plot."""
        try:
            data = self.inspector.get_cell_data(trial, stage)
            stats = self.inspector.get_cell_stats(trial, stage)
            stage_name = self.inspector.pipeline[stage].name
            row_label = self.inspector.row_label
            
            # Apply epoch if set
            start, end, tm_type = self.epoch_widget.get_epoch()   ## PASS START & END WITH DATA
            if start is not None and end is not None:
                logging.debug(f"start: {start}, end: {end}, tm_type: {tm_type}")
            #if end > 0:
            #   data = data[start:end]
            
            
                self.plot_widget.plot_single_epoched(  
                    data, 
                    title=f"{row_label} {trial + 1} → {stage_name}",
                    sample_rate=self.inspector.sample_rate,
                    start=start,
                    end=end,
                    tm_type= tm_type
                )
            else:
                self.plot_widget.plot_single(
                    data,
                    title=f"{row_label} {trial + 1} → {stage_name}",
                    sample_rate=self.inspector.sample_rate,
                    
                )
            
            self._update_status(
                f"{row_label} {trial + 1} | {stage_name} | "
                f"μ={stats['mean']:.3f} σ={stats['std']:.3f} | "
                f"Samples: {stats['length']} | Double-click to inspect values"
            )
        except Exception as e:
            self._update_status(f"Error: {e}")
    
    def _on_cell_double_clicked(self, trial: int, stage: int):
        """Handle cell double-click - open array inspector dialog."""
        from .cell_dialog import CellDialog
        
        dialog = CellDialog(self.inspector, trial, stage, parent=self)
        dialog.exec()
    
    def _on_selection_changed(self, selected: list[tuple[int, int]]):
        """Handle selection change - update button states and auto-average if multiple rows selected."""
        self.selected_cells = selected
        has_multiple = len(selected) > 1
        self.overlay_button.setEnabled(has_multiple)
        
        # Auto-average when multiple cells are selected in the same stage
        if has_multiple:
            # Check if all selected cells are in the same stage (column)
            stages = set(stage for _, stage in selected)
            if len(stages) == 1:
                stage = list(stages)[0]
                trials = [trial for trial, _ in selected]
                self._show_averaged_data(trials, stage)
    
    def _show_averaged_data(self, trials: list[int], stage: int):
        """Show averaged data for selected trials at a given stage."""
        avg_data = self.inspector.get_averaged_data(trials, stage)
        stage_name = self.inspector.pipeline[stage].name
        row_label = self.inspector.row_label
        
        # Apply epoch if set
        start, end, tm_type = self.epoch_widget.get_epoch()
        if end > 0:
            avg_data = avg_data[start:end]
        
        self.plot_widget.plot_single(
            avg_data,
            title=f"Average of {len(trials)} {row_label}s → {stage_name}",
            sample_rate=self.inspector.sample_rate
        )
        
        self._update_status(
            f"Averaged {len(trials)} {row_label}s | {stage_name} | "
            f"μ={avg_data.mean():.3f} σ={avg_data.std():.3f}"
        )
    
    def _on_average_all_rows(self):
        """Average all rows for a selected stage (or first stage if none selected)."""
        # Determine which stage to average
        if self.selected_cells:
            # Use the stage from the last selected cell
            stage = self.selected_cells[-1][1]
        else:
            # Default to first stage (Raw)
            stage = 0
        
        # Get all trials
        all_trials = list(range(self.inspector.n_trials))
        self._show_averaged_data(all_trials, stage)
    
    def _on_overlay_selected(self):
        """Overlay all selected cells on the plot."""
        if not self.selected_cells:
            return
        
        self.plot_widget.clear()
        
        start, end, tm_type = self.epoch_widget.get_epoch()
        row_label = self.inspector.row_label
        
        for trial, stage in self.selected_cells:
            data = self.inspector.get_cell_data(trial, stage)
            if end > 0:
                data = data[start:end]
            
            stage_name = self.inspector.pipeline[stage].name
            # Use custom row name if available
            if self.inspector.row_names and trial < len(self.inspector.row_names):
                row_name = self.inspector.row_names[trial]
            else:
                row_name = f"{row_label}{trial + 1}"
            
            self.plot_widget.add_trace(
                data,
                name=f"{row_name}-{stage_name}",
                sample_rate=self.inspector.sample_rate
            )
        
        self._update_status(f"Overlaying {len(self.selected_cells)} traces")
    
    def _on_epoch_changed(self, start: int, end: int):
        """Handle epoch change - update current plot."""
        # Re-plot current selection with new epoch
        if self.selected_cells:
            if len(self.selected_cells) == 1:
                trial, stage = self.selected_cells[0]
                self._on_cell_clicked(trial, stage)
    
    def _on_reset_view(self):
        """Reset epoch to full trace."""
        self.epoch_widget.reset()
        self.plot_widget.reset_view()
        self._update_status("View reset to full trace")
        
    def _reset_all(self):
        """Reset all widgets."""
        self.epoch_widget.reset()
        self.plot_widget.reset_view()
        self.selected_cells = []
        self.grid_widget.populate(self.inspector)
        
        
        self._update_status("Full view reset")
        


def launch_app(inspector: "Inspector"):
    """
    Launch the TSGeneral application.
    
    Args:
        inspector: Configured Inspector instance
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("TSGeneral")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("TSGeneral")
    
    window = MainWindow(inspector)
    window.show()
    
    sys.exit(app.exec())
