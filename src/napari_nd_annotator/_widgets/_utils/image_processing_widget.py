import sys

import napari
from napari.utils.notifications import notification_manager
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QAction, QPlainTextEdit
from qtpy.QtCore import QRegularExpression, Qt, Signal, QObject, QThread
from qtpy.QtGui import QTextCharFormat, QFont, QSyntaxHighlighter, QColor
from qtpy import QtCore, QtGui
import numpy as np
import skimage
import warnings
import keyword

from napari_nd_annotator._widgets._utils import ProgressWidget


def execute_script(script, image, other_locals=None):
    if other_locals is None:
        other_locals = dict()
    locals = {"image": image} | other_locals
    exec(script, {"np": np, "skimage": skimage}, locals)
    if "features" in locals:
        features = locals["features"]
    else:
        features = None
        warnings.warn("The output should be stored in a variable called 'features'")
    return features


class ScriptWorker(QObject):
    done = Signal("PyQt_PyObject")
    script = None
    image = None

    def run(self):
        if self.script is None:
            raise ValueError("No script was set!")
        if self.image is None:
            raise ValueError("'image' was None!")
        try:
            features = execute_script(self.script, self.image)
            self.done.emit(features)
        except Exception as e:
            self.done.emit(None)
            raise e



class PythonHighlighter(QSyntaxHighlighter):
    class HighlightingRule:
        pattern = QRegularExpression()
        format = QTextCharFormat()
    highlightingRules = []
    commentFormat = QTextCharFormat()
    keywordFormat = QTextCharFormat()
    classFormat = QTextCharFormat()
    quotationFormat = QTextCharFormat()
    singleLineCommentFormat = QTextCharFormat()
    tripleQuotationFormat = QTextCharFormat()
    functionFormat = QTextCharFormat()
    numericFormat = QTextCharFormat()
    
    def __init__(self, parent):
        super().__init__(parent)
        self.keywordFormat.setForeground(QColor("darkorange"))
        self.keywordFormat.setFontWeight(QFont.Bold)
        keywordPatterns = ["\\b%s\\b" % kw for kw in keyword.kwlist]
        for pattern in keywordPatterns:
            rule = self.HighlightingRule()
            rule.pattern = QRegularExpression(pattern)
            rule.format = self.keywordFormat
            self.highlightingRules.append(rule)

        rule = self.HighlightingRule()
        self.numericFormat.setForeground(QColor("deepskyblue"))
        rule.pattern = QRegularExpression("(0((b[01]+)|(O[0-7]+)|(x[0-9A-Fa-f]+)))|([0-9]*\\.?[0-9]*j?)")
        rule.format = self.numericFormat
        self.highlightingRules.append(rule)

        rule = self.HighlightingRule()
        self.functionFormat.setForeground(QColor("gold"))
        rule.pattern = QRegularExpression("\\b[a-z][A-Za-z0-9_]*(?=\\()")
        rule.format = self.functionFormat
        self.highlightingRules.append(rule)

        rule = self.HighlightingRule()
        self.classFormat.setForeground(QColor("lightblue"))
        rule.pattern = QRegularExpression("\\b[A-Z][A-Za-z0-9_]*(?=\\()")
        rule.format = self.classFormat
        self.highlightingRules.append(rule)

        rule = self.HighlightingRule()
        self.quotationFormat.setForeground(Qt.darkGreen)
        rule.pattern = QRegularExpression("([\"'])(?:(?=(\\\\?))\\2.)*?\\1")
        rule.format = self.quotationFormat
        self.highlightingRules.append(rule)

        rule = self.HighlightingRule()
        self.commentFormat.setForeground(Qt.gray)
        rule.pattern = QRegularExpression("#.*")
        rule.format = self.commentFormat
        self.highlightingRules.append(rule)

    def highlightBlock(self, text):
        for rule in self.highlightingRules:
            matchIterator = rule.pattern.globalMatch(text)
            while matchIterator.hasNext():
                match = matchIterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), rule.format)


class CodeEditor(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("font-family:'Courier New'; background-color: black")

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key_Tab:
            self.insertPlainText("    ")
        else:
            super().keyPressEvent(event)


class ImageProcessingWidget(QWidget):
    def __init__(self, image, viewer: napari.Viewer, editor_key="img_proc_script", parent=None):
        super().__init__(parent=parent)
        self.viewer = viewer
        self._editor_key = editor_key
        self.progress_dialog = ProgressWidget(message="Calculating feature, please wait...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            viewer.window.qt_viewer.window().destroyed.connect(self.on_destroy)
        self.script_thread = QThread()
        self.script_worker = ScriptWorker()
        self.script_worker.moveToThread(self.script_thread)
        self.script_worker.done.connect(self.script_thread.quit)
        self.script_worker.done.connect(lambda _: self.progress_dialog.setVisible(False))
        self.script_worker.done.connect(lambda: self.run_button.setEnabled(True))
        self.script_worker.done.connect(lambda: self.try_button.setEnabled(True))
        self.script_thread.started.connect(self.script_worker.run)
        self.script_thread.finished.connect(lambda: self.progress_dialog.setVisible(False))
        layout = QVBoxLayout()
        self.text_settings = QtCore.QSettings("BIOMAG", "Annotation Toolbox")
        self.text_edit = CodeEditor()
        self.text_edit.document().setPlainText(self.text_settings.value(self._editor_key, ""))
        self.highlighter = PythonHighlighter(self.text_edit.document())
        font_size = self.text_settings.value("script_font_size")
        if font_size:
            font = self.text_edit.font()
            font.setPointSize(font_size)
            self.text_edit.setFont(font)
        run_action_enter = QAction(self.text_edit)
        run_action_enter.setAutoRepeat(False)
        run_action_enter.setShortcut("Ctrl+Enter")
        run_action_enter.triggered.connect(self.execute)
        self.text_edit.addAction(run_action_enter)

        run_action_return = QAction(self.text_edit)
        run_action_return.setAutoRepeat(False)
        run_action_return.setShortcut("Ctrl+Return")
        run_action_return.triggered.connect(self.execute)
        self.text_edit.addAction(run_action_return)

        increase_fontsize_action = QAction(self.text_edit)
        increase_fontsize_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL+QtCore.Qt.Key_Plus))
        increase_fontsize_action.triggered.connect(self.increase_font_size)
        self.text_edit.addAction(increase_fontsize_action)

        decrease_fontsize_action = QAction(self.text_edit)
        decrease_fontsize_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL+QtCore.Qt.Key_Minus))
        decrease_fontsize_action.triggered.connect(self.decrease_font_size)
        self.text_edit.addAction(decrease_fontsize_action)
        layout.addWidget(self.text_edit)

        buttons_layout = QHBoxLayout()
        self.run_button = QPushButton("Set")
        self.run_button.clicked.connect(self.execute)
        buttons_layout.addWidget(self.run_button)
        self.try_button = QPushButton("Try")
        self.try_button.clicked.connect(self.try_code)
        buttons_layout.addWidget(self.try_button)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
        self.image = image
        self.features = image

    def run_script(self):
        script = self.text_edit.document().toPlainText()
        self.text_settings.setValue(self._editor_key, script)
        if self.image is None:
            warnings.warn("image is None")
            return None
        self.run_button.setEnabled(False)
        self.try_button.setEnabled(False)
        self.script_worker.script = script
        self.script_worker.image = self.image.copy() if self.image is not None else None
        self.progress_dialog.setVisible(True)
        self.script_thread.start()

    def set_features(self, features):
        if features is None:
            return
        self.features = features.astype(float)
        self.script_worker.done.disconnect(self.set_features)

    def execute(self):
        self.script_worker.done.connect(self.set_features)
        self.run_script()

    def display_features(self, features):
        if features is not None:
            if "Feature map" not in self.viewer.layers:
                self.viewer.add_image(
                    features,
                    name="Feature map"
                )
            else:
                self.viewer.layers["Feature map"].data = features

    def try_code(self):
        features = self.execute_script()
        self.display_features(features)

    def execute_script(self, image=None):
        if image is None:
            image = self.image
        script = self.text_edit.document().toPlainText()
        self.text_settings.setValue(self._editor_key, script)
        return execute_script(script, image)

    def calculate_features(self, image):
        return execute_script()

    def increase_font_size(self):
        font = self.text_edit.font()
        font.setPointSize(font.pointSize()+1)
        self.text_edit.setFont(font)
        self.text_settings.setValue("script_font_size", font.pointSize())

    def decrease_font_size(self):
        font = self.text_edit.font()
        font.setPointSize(font.pointSize()-1)
        self.text_edit.setFont(font)
        self.text_settings.setValue("script_font_size", font.pointSize())

    def on_destroy(self):
        try:
            self.progress_dialog.close()
        except:
            ...
