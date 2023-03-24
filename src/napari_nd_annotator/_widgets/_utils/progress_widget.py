from qtpy.QtWidgets import QWidget, QProgressBar, QLabel, QVBoxLayout
from qtpy.QtCore import Qt
from qtpy import QtGui

class ProgressWidget(QWidget):
    def __init__(self, parent=None, min_value=0, max_value=0, value=0, message="running...", visible=False, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.drag_start_x = None
        self.drag_start_y = None
        layout = QVBoxLayout()
        self.label = QLabel(message)
        layout.addWidget(self.label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(min_value)
        self.progress_bar.setMaximum(max_value)
        self.progress_bar.setValue(value)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint)
        self.setVisible(visible)
        layout.addWidget(self.progress_bar)
        layout.addStretch()
        self.setLayout(layout)

    def reset(self):
        self.progress_bar.reset()

    def setMinimum(self, value):
        self.progress_bar.setMinimum(value)

    def setMaximum(self, value):
        self.progress_bar.setMaximum(value)

    def setValue(self, value):
        self.progress_bar.setValue(value)

    def setText(self, text):
        self.label.setText(text)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.drag_start_x = event.x()
        self.drag_start_y = event.y()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self.move(event.globalX() - self.drag_start_x, event.globalY()-self.drag_start_y)