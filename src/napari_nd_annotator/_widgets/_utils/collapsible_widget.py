from typing import List, Optional

from magicclass.widgets import CollapsibleContainer
from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout, QSizePolicy
from qtpy.QtCore import Signal


class CollapsibleWidget(QWidget):
    expanded = Signal()
    collapsed = Signal()
    expansion_changed = Signal(bool)
    def __init__(self, text="", parent=None):
        super().__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self._collapsed = True
        self._text = text
        self.collapse_button = QPushButton(self._prefix() + text, parent=self)
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
        prev_val = self.isCollapsed()
        self._collapsed = bool(is_collapsed)
        self.collapse_button.setText(self._prefix() + self._text)
        self.content_widget.setVisible(not self._collapsed)
        if prev_val != self.isCollapsed():
            if self.isCollapsed():
                self.collapsed.emit()
            else:
                self.expanded.emit()
            self.expansion_changed.emit(not self.isCollapsed())

    def isCollapsed(self):
        return self._collapsed

    def collapse(self):
        self.setCollapsed(True)

    def expand(self):
        self.setCollapsed(False)

    def _prefix(self):
        return "\N{BLACK MEDIUM RIGHT-POINTING TRIANGLE} " if self._collapsed else "\N{BLACK MEDIUM DOWN-POINTING TRIANGLE} "


class CollapsibleWidgetGroup:
    def __init__(self, widget_list: Optional[List[CollapsibleWidget]] = None):
        self._widget_list = []
        self._handlers = dict()
        if widget_list is None:
            widget_list = []
        for widget in widget_list:
            if type(widget) != CollapsibleWidget:
                raise TypeError("%s is not CollapsibleWidget" % str(widget))
            self.addItem(widget)

    def addItem(self, widget: CollapsibleWidget):
        if widget in self._widget_list:
            raise ValueError("%s widget already in group" % str(widget))
        self._widget_list.append(widget)
        self._handlers[widget] = self._widget_expanded_handler(widget)
        widget.expanded.connect(self._handlers[widget])

    def removeItem(self, widget: CollapsibleWidget):
        if widget not in self._widget_list:
            return
        self._widget_list.remove(widget)
        widget.expanded.disconnect(self._handlers[widget])
        del self._handlers[widget]

    def _widget_expanded_handler(self, widget: CollapsibleWidget):
        def handler():
            for w2 in self._widget_list:
                if w2 != widget:
                    w2.collapse()
        return handler


class CollapsibleContainerGroup:
    def __init__(self, container_list: Optional[List[CollapsibleContainer]] = None):
        self._container_list = []
        self._handlers = dict()
        if container_list is None:
            container_list = []
        for container in container_list:
            if type(container) != CollapsibleContainer:
                raise TypeError("%s is not CollapsibleContainer" % str(container))
            self.addItem(container)

    def addItem(self, container: CollapsibleContainer):
        if container in self._container_list:
            return
        self._container_list.append(container)
        self._handlers[container] = self._container_expanded_handler(container)
        container._widget._expand_btn.clicked.connect(self._handlers[container])

    def removeItem(self, container: CollapsibleContainer):
        if container not in self._container_list:
            return
        self._container_list.remove(container)
        container._widget._expand_btn.clicked.disconnect(self._handlers[container])
        del self._handlers[container]

    def _container_expanded_handler(self, container: CollapsibleContainer):
        def handler(state: bool):
            if not state:
                return
            for c2 in self._container_list:
                if c2 != container:
                    c2._widget._collapse()
                    c2._widget._expand_btn.setChecked(False)
        return handler

    def __iter__(self):
        return iter(self._container_list)

def correct_container_size(container):
    if not container.collapsed:
        container.collapsed = True
        container.collapsed = False