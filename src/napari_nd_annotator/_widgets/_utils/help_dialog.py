from qtpy.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QAbstractItemView, QSizePolicy
from qtpy.QtCore import Qt
from .shortcuts import help_rows


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        rows = help_rows()
        table_widget = QTableWidget()
        table_widget.setRowCount(len(rows))
        table_widget.setColumnCount(2)
        table_widget.setHorizontalHeaderLabels(["Action", "Shortcut"])
        for i, (action, shortcut) in enumerate(rows):
            table_widget.setItem(i, 0, QTableWidgetItem(action))
            table_widget.setItem(i, 1, QTableWidgetItem(shortcut))
        table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table_widget.setFocusPolicy(Qt.NoFocus)
        table_widget.setSelectionMode(QAbstractItemView.NoSelection)
        table_widget.verticalHeader().hide()
        table_widget.resizeColumnsToContents()
        table_widget.setWordWrap(True)
        table_widget.horizontalHeader().sectionResized.connect(table_widget.resizeRowsToContents)
        layout = QVBoxLayout()
        layout.addWidget(table_widget)
        self.setLayout(layout)
        dialogWidth = table_widget.horizontalHeader().length() + 50
        dialogHeight = table_widget.verticalHeader().length() + 24
        self.setFixedSize(dialogWidth, dialogHeight)
