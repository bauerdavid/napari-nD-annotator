from qtpy.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QAbstractItemView, QSizePolicy
from qtpy.QtCore import Qt
shortcuts = {
    "Ctrl+I": "Interpolate",
    "E/Shift+Wheel " + u"\u2191": "Increment selected label",
    "Q/Shift+Wheel " + u"\u2193": "Decrement selected label",
    "A/Ctrl+Wheel " + u"\u2191": "Previous slice",
    "D/Ctrl+Wheel " + u"\u2193": "Next slice",
    "W/Alt+Wheel " + u"\u2191": "Increase paint brush size",
    "S/Alt+Wheel " + u"\u2193": "Decrease paint brush size",
    "Ctrl+Tab": "Jump from \"Anchors\" layer to the labels layer, or vice versa",
    "Ctrl+[1-10]": "Jump to the layer at the selected index"
}


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        table_widget = QTableWidget()
        table_widget.setRowCount(len(shortcuts))
        table_widget.setColumnCount(2)
        table_widget.setHorizontalHeaderLabels(["Action", "Shortcut"])
        for i, (shortcut, action) in enumerate(shortcuts.items()):
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
