from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout, QSizePolicy


class CollapsibleWidget(QWidget):
    def __init__(self, text="", parent=None):
        super().__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout()
        self._collapsed = True
        self._text = text
        self.collapse_button = QPushButton("\N{BLACK RIGHT-POINTING TRIANGLE} " + text, parent=self)
        self.collapse_button.setSizePolicy(QSizePolicy.Expanding, self.collapse_button.sizePolicy().verticalPolicy())
        self.collapse_button.clicked.connect(lambda: self.setCollapsed(not self._collapsed))
        self.collapse_button.setStyleSheet("Text-align:left")
        layout.addWidget(self.collapse_button)

        self.content_widget = QWidget(parent=self)
        self.content_widget.setVisible(not self._collapsed)
        layout.addWidget(self.content_widget)

        super().setLayout(layout)

    def setLayout(self, QLayout):
        self.content_widget.setLayout(QLayout)

    def layout(self):
        return self.content_widget.layout()

    def setCollapsed(self, is_collapsed):
        self._collapsed = bool(is_collapsed)
        self.collapse_button.setText(self._prefix() + self._text)
        self.content_widget.setVisible(not self._collapsed)

    def isCollapsed(self, is_collapsed):
        return self._collapsed

    def collapse(self):
        self.setCollapsed(True)

    def expand(self):
        self.setCollapsed(False)

    def _prefix(self):
        return "\N{BLACK MEDIUM RIGHT-POINTING TRIANGLE} " if self._collapsed else "\N{BLACK MEDIUM DOWN-POINTING TRIANGLE} "
