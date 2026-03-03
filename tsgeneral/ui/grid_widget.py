"""
Pipeline Grid Widget - displays trials × stages matrix.
"""

from typing import TYPE_CHECKING, Optional

from PySide6.QtWidgets import (
    QWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QHeaderView, QAbstractItemView, QMenu
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QBrush

if TYPE_CHECKING:
    from ..inspector import Inspector


class CellWidget(QTableWidgetItem):
    """
    Custom table cell that displays summary stats for a trial/stage.
    """
    
    def __init__(self, stats: dict, trial: int, stage: int):
        super().__init__()
        self.trial = trial
        self.stage = stage
        self.stats = stats
        
        # Format display text
        self.setText(self._format_stats())
        self.setToolTip(self._format_tooltip())
        
        # Style
        self.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def _format_stats(self) -> str:
        """Format stats for cell display."""
        return f"μ={self.stats['mean']:.2f}\nσ={self.stats['std']:.2f}\nMin={self.stats['min']:.2f}\nMax={self.stats['max']:.2f}"
    
    def _format_tooltip(self) -> str:
        """Format detailed tooltip."""
        return (
            f"Trial {self.trial + 1}\n"
            f"Mean: {self.stats['mean']:.4f}\n"
            f"Std: {self.stats['std']:.4f}\n"
            f"Min: {self.stats['min']:.4f}\n"
            f"Max: {self.stats['max']:.4f}\n"
            f"Samples: {self.stats['length']}"
        )


class PipelineGridWidget(QWidget):
    """
    Grid widget showing trials (rows) × pipeline stages (columns).
    
    Signals:
        cell_clicked: Emitted when a cell is clicked (trial, stage)
        selection_changed: Emitted when selection changes (list of (trial, stage))
    """
    
    cell_clicked = Signal(int, int)  # trial, stage
    cell_double_clicked = Signal(int, int)  # trial, stage - for opening array inspector
    selection_changed = Signal(list)  # list of (trial, stage)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.inspector: Optional["Inspector"] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the table widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.table = QTableWidget()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        
        # Connect signals
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        
        # Style
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        
        # Context menu
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        
        layout.addWidget(self.table)
    
    def populate(self, inspector: "Inspector"):
        """
        Populate the grid with data from an Inspector.
        
        Args:
            inspector: Inspector instance with loaded data
        """
        self.inspector = inspector
        
        n_trials = inspector.n_trials
        n_stages = inspector.n_stages
        
        self.table.clear()
        self.table.setRowCount(n_trials)
        self.table.setColumnCount(n_stages)
        
        # Set headers
        stage_names = [stage.name for stage in inspector.pipeline]
        self.table.setHorizontalHeaderLabels(stage_names)
        
        # Use custom row names if provided, otherwise generate from row_label
        if inspector.row_names is not None:
            trial_labels = inspector.row_names[:n_trials]
        else:
            trial_labels = [f"{inspector.row_label} {i + 1}" for i in range(n_trials)]
        self.table.setVerticalHeaderLabels(trial_labels)
        
        # Populate cells
        for trial in range(n_trials):
            for stage in range(n_stages):
                stats = inspector.get_cell_stats(trial, stage)
                cell = CellWidget(stats, trial, stage)
                self.table.setItem(trial, stage, cell)
        
        # Adjust row heights
        self.table.resizeRowsToContents()
    
    def _on_cell_clicked(self, row: int, col: int):
        """Handle cell click."""
        self.cell_clicked.emit(row, col)
    
    def _on_cell_double_clicked(self, row: int, col: int):
        """Handle cell double-click - open array inspector."""
        self.cell_double_clicked.emit(row, col)
    
    def _on_selection_changed(self):
        """Handle selection change."""
        selected = []
        for item in self.table.selectedItems():
            if isinstance(item, CellWidget):
                selected.append((item.trial, item.stage))
        self.selection_changed.emit(selected)
    
    def _show_context_menu(self, pos):
        """Show context menu for selected cells."""
        menu = QMenu(self)
        
        selected = self.get_selected_cells()
        
        if len(selected) == 1:
            # Single cell selected
            inspect_action = menu.addAction("🔍 Inspect Array Values")
            inspect_action.triggered.connect(lambda: self.cell_double_clicked.emit(selected[0][0], selected[0][1]))
            
            plot_action = menu.addAction("📈 Plot This Cell")
            plot_action.triggered.connect(lambda: self.cell_clicked.emit(selected[0][0], selected[0][1]))
        
        elif len(selected) > 1:
            # Multiple cells selected
            # Check if same column
            stages = set(s for _, s in selected)
            if len(stages) == 1:
                menu.addAction(f"📊 Averaging {len(selected)} rows (auto)")
            else:
                overlay_action = menu.addAction(f"📈 Overlay {len(selected)} cells")
                overlay_action.triggered.connect(self._trigger_overlay)
        
        if selected:
            menu.addSeparator()
            select_col_action = menu.addAction("Select Entire Column")
            select_col_action.triggered.connect(lambda: self._select_column(selected[0][1]))
            
            select_row_action = menu.addAction("Select Entire Row")  
            select_row_action.triggered.connect(lambda: self._select_row(selected[0][0]))
        
        menu.exec(self.table.mapToGlobal(pos))
    
    def _trigger_overlay(self):
        """Trigger overlay via selection change (handled by main window)."""
        # The main window will handle this via the Overlay Selected button
        pass
    
    def _select_column(self, col: int):
        """Select all cells in a column."""
        self.table.clearSelection()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, col)
            if item:
                item.setSelected(True)
    
    def _select_row(self, row: int):
        """Select all cells in a row."""
        self.table.clearSelection()
        for col in range(self.table.columnCount()):
            item = self.table.item(row, col)
            if item:
                item.setSelected(True)
    
    def get_selected_cells(self) -> list[tuple[int, int]]:
        """Get list of selected (trial, stage) tuples."""
        selected = []
        for item in self.table.selectedItems():
            if isinstance(item, CellWidget):
                selected.append((item.trial, item.stage))
        return selected
