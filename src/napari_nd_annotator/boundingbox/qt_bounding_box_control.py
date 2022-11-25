# A copy of napari._qt.layer_controls.qt_shapes_controls
from typing import Iterable

import napari
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.utils.action_manager import action_manager
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QButtonGroup, QGridLayout, QLabel, QHBoxLayout, QCheckBox, QComboBox
from napari._qt.widgets._slider_compat import QDoubleSlider, QSlider
from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
import numpy as np
from napari._qt.utils import qt_signals_blocked, disable_with_opacity
from napari._qt.widgets.qt_mode_buttons import QtModeRadioButton, QtModePushButton
from napari.utils.events import disconnect_events
from napari.utils.interactions import Shortcut
from napari.utils.translations import trans

from ._bounding_box_constants import Mode

class QtBoundingBoxControls(QtLayerControls):
    # TODO: review comments
    """Qt view and controls for the napari BoundingBoxLayer layer.

    Parameters
    ----------
    layer : napari.layers.BoundingBoxLayer
        An instance of a napari BoundingBoxLayer layer.

    Attributes
    ----------
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group for bounding boxes layer modes
        (SELECT, DIRECT, PAN_ZOOM, ADD_RECTANGLE, ADD_ELLIPSE, ADD_LINE,
        ADD_PATH, ADD_POLYGON, VERTEX_INSERT, VERTEX_REMOVE).
    delete_button : qtpy.QtWidgets.QtModePushButton
        Button to delete selected bounding boxes
    edgeColorSwatch : qtpy.QtWidgets.QFrame
        Thumbnail display of points edge color.
    edgeComboBox : qtpy.QtWidgets.QComboBox
        Drop down list allowing user to set edge color of points.
    ellipse_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add ellipses to bounding boxes layer.
    faceColorSwatch : qtpy.QtWidgets.QFrame
        Thumbnail display of points face color.
    faceComboBox : qtpy.QtWidgets.QComboBox
        Drop down list allowing user to set face color of points.
    grid_layout : qtpy.QtWidgets.QGridLayout
        Layout of Qt widget controls for the layer.
    layer : napari.layers.BoundingBoxLayer
        An instance of a napari BoundingBoxLayer layer.
    panzoom_button : qtpy.QtWidgets.QtModeRadioButton
        Button to pan/zoom bounding boxes layer.
    bounding_box_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add rectangles to bounding boxes layer.
    select_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select bounding boxes.
    widthSlider : qtpy.QtWidgets.QSlider
        Slider controlling line edge width of bounding boxes.

    Raises
    ------
    ValueError
        Raise error if bounding boxes mode is not recognized.
    """

    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.mode.connect(self._on_mode_change)
        self.layer.events.size_mode.connect(self._on_size_mode_change)
        self.layer.events.size_multiplier.connect(self._on_size_multiplier_change)
        self.layer.events.size_constant.connect(self._on_size_constant_change)
        self.layer.events.edge_width.connect(self._on_edge_width_change)
        self.layer.events.current_edge_color.connect(
            self._on_current_edge_color_change
        )
        self.layer.events.current_face_color.connect(
            self._on_current_face_color_change
        )
        self.layer.events.editable.connect(self._on_editable_change)
        self.layer.text.events.visible.connect(self._on_text_visibility_change)

        sld = QSlider(Qt.Horizontal)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(40)
        sld.setSingleStep(1)
        value = self.layer.current_edge_width
        if isinstance(value, Iterable):
            if isinstance(value, list):
                value = np.asarray(value)
            value = value.mean()
        sld.setValue(int(value))
        sld.valueChanged.connect(self.changeWidth)
        self.widthSlider = sld

        sld = QSlider(Qt.Horizontal)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(2)
        sld.setMaximum(200)
        sld.setSingleStep(1)
        value = self.layer.text.size
        if isinstance(value, Iterable):
            if isinstance(value, list):
                value = np.asarray(value)
            value = value.mean()
        sld.setValue(int(value))
        sld.valueChanged.connect(self.changeTextSize)
        self.textSlider = sld

        def _radio_button(
            parent,
            btn_name,
            mode,
            action_name,
            extra_tooltip_text='',
            **kwargs,
        ):
            """
            Convenience local function to create a RadioButton and bind it to
            an action at the same time.

            Parameters
            ----------
            parent : Any
                Parent of the generated QtModeRadioButton
            btn_name : str
                name fo the button
            mode : Enum
                Value Associated to current button
            action_name : str
                Action triggered when button pressed
            extra_tooltip_text : str
                Text you want added after the automatic tooltip set by the
                action manager
            **kwargs:
                Passed to QtModeRadioButton

            Returns
            -------
            button: QtModeRadioButton
                button bound (or that will be bound to) to action `action_name`

            Notes
            -----
            When shortcuts are modifed/added/removed via the action manager, the
            tooltip will be updated to reflect the new shortcut.
            """
            action_name = 'napari:' + action_name
            btn = QtModeRadioButton(parent, btn_name, mode, **kwargs)
            '''action_manager.bind_button(
                action_name,
                btn,
                extra_tooltip_text='',
            )'''
            return btn

        self.select_button = _radio_button(
            layer, 'select', Mode.SELECT, "activate_bb_select_mode"
        )

        self.panzoom_button = _radio_button(
            layer,
            'zoom',
            Mode.PAN_ZOOM,
            "activate_bb_pan_zoom_mode",
            extra_tooltip_text=trans._('(or hold Space)'),
            checked=True,
        )

        self.bounding_box_button = _radio_button(
            layer,
            'rectangle',
            Mode.ADD_BOUNDING_BOX,
            "activate_add_bb_mode",
        )

        self.delete_button = QtModePushButton(
            layer,
            'delete_shape',
            slot=self.layer.remove_selected,
            tooltip=trans._(
                "Delete selected bounding boxes ({shortcut})",
                shortcut=Shortcut('Backspace').platform,
            ),
        )

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.select_button)
        self.button_group.addButton(self.panzoom_button)
        self.button_group.addButton(self.bounding_box_button)

        button_row = QHBoxLayout()
        button_row.addWidget(self.delete_button)
        button_row.addWidget(self.select_button)
        button_row.addWidget(self.panzoom_button)
        button_row.addWidget(self.bounding_box_button)
        button_row.setContentsMargins(0, 0, 0, 5)
        button_row.setSpacing(4)

        bb_size_mode_combobox = QComboBox()
        bb_size_mode_combobox.addItem("average")
        bb_size_mode_combobox.addItem("constant")
        bb_size_mode_combobox.activated[str].connect(self.changeSizeMode)
        self.bb_size_mode_combobox = bb_size_mode_combobox

        bb_size_mult_slider = QDoubleSlider(Qt.Horizontal, parent=self)
        bb_size_mult_slider.setFocusPolicy(Qt.NoFocus)
        bb_size_mult_slider.setMinimum(0.1)
        bb_size_mult_slider.setMaximum(10)
        bb_size_mult_slider.setSingleStep(0.1)
        bb_size_mult_slider.valueChanged.connect(self.changeSizeMultiplier)
        self.bb_size_mult_slider = bb_size_mult_slider
        self.bb_size_mult_label = QLabel(trans._('size multiplier:'))
        self._on_size_multiplier_change()

        bb_size_const_slider = QSlider(Qt.Horizontal)
        bb_size_const_slider.setFocusPolicy(Qt.NoFocus)
        bb_size_const_slider.setMinimum(1)
        bb_size_const_slider.setMaximum(100)
        bb_size_const_slider.setSingleStep(1)
        bb_size_const_slider.valueChanged.connect(self.changeSizeConst)
        self.bb_size_const_slider = bb_size_const_slider
        self.bb_size_const_label = QLabel(trans._('size constant: '))
        self._on_size_constant_change()
        self._on_size_mode_change()


        self.faceColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_face_color,
            tooltip=trans._('click to set current face color'),
        )
        self._on_current_face_color_change()
        self.edgeColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_edge_color,
            tooltip=trans._('click to set current edge color'),
        )
        self._on_current_edge_color_change()

        self.textColorEdit = QColorSwatchEdit(
            initial_color=self.layer.text.color,
            tooltip=trans._('click to set current text color'),
        )
        self._on_current_text_color_change()

        self.faceColorEdit.color_changed.connect(self.changeFaceColor)
        self.edgeColorEdit.color_changed.connect(self.changeEdgeColor)
        self.textColorEdit.color_changed.connect(self.changeTextColor)

        text_disp_cb = QCheckBox()
        text_disp_cb.setToolTip(trans._('toggle text visibility'))
        text_disp_cb.setChecked(self.layer.text.visible)
        text_disp_cb.stateChanged.connect(self.change_text_visibility)
        self.textDispCheckBox = text_disp_cb

        if napari.__version__ == "0.4.15":
            # grid_layout created in QtLayerControls
            # addWidget(widget, row, column, [row_span, column_span])
            self.grid_layout.addLayout(button_row, 0, 1)
            self.grid_layout.addWidget(QLabel(trans._('opacity:')), 1, 0)
            self.grid_layout.addWidget(self.opacitySlider, 1, 1)
            self.grid_layout.addWidget(QLabel(trans._('edge width:')), 2, 0)
            self.grid_layout.addWidget(self.widthSlider, 2, 1)
            self.grid_layout.addWidget(QLabel(trans._('blending:')), 3, 0)
            self.grid_layout.addWidget(self.blendComboBox, 3, 1)
            self.grid_layout.addWidget(QLabel(trans._('size mode:')), 4, 0)
            self.grid_layout.addWidget(self.bb_size_mode_combobox, 4, 1)
            self.grid_layout.addWidget(self.bb_size_mult_label, 5, 0)
            self.grid_layout.addWidget(self.bb_size_mult_slider, 5, 1)
            self.grid_layout.addWidget(self.bb_size_const_label, 6, 0)
            self.grid_layout.addWidget(self.bb_size_const_slider, 6, 1)
            self.grid_layout.addWidget(QLabel(trans._('face color:')), 7, 0)
            self.grid_layout.addWidget(self.faceColorEdit, 7, 1)
            self.grid_layout.addWidget(QLabel(trans._('edge color:')), 8, 0)
            self.grid_layout.addWidget(self.edgeColorEdit, 8, 1)
            self.grid_layout.addWidget(QLabel(trans._('display text:')), 9, 0)
            self.grid_layout.addWidget(self.textDispCheckBox, 9, 1)
            self.grid_layout.addWidget(QLabel(trans._('text color:')), 10, 0)
            self.grid_layout.addWidget(self.textColorEdit, 10, 1)
            self.grid_layout.addWidget(QLabel(trans._('text size:')), 11, 0)
            self.grid_layout.addWidget(self.textSlider, 11, 1)
            self.grid_layout.setRowStretch(11, 1)
            self.grid_layout.setColumnStretch(1, 1)
            self.grid_layout.setSpacing(4)
        else:
            self.layout().addRow(button_row)
            self.layout().addRow(trans._('opacity:'), self.opacitySlider)
            self.layout().addRow(trans._('edge width:'), self.widthSlider)
            self.layout().addRow(trans._('blending:'), self.blendComboBox)
            self.layout().addRow(trans._('size mode:'), self.bb_size_mode_combobox)
            self.layout().addRow(self.bb_size_mult_label, self.bb_size_mult_slider)
            self.layout().addRow(self.bb_size_const_label, self.bb_size_const_slider)
            self.layout().addRow(trans._('face color:'), self.faceColorEdit)
            self.layout().addRow(trans._('edge color:'), self.edgeColorEdit)
            self.layout().addRow(trans._('display text:'), self.textDispCheckBox)
            self.layout().addRow(trans._('text color:'), self.textColorEdit)
            self.layout().addRow(trans._('text size:'), self.textSlider)

    def _on_mode_change(self, event):
        """Update ticks in checkbox widgets when bounding boxes layer mode changed.

        Available modes for bounding boxes layer are:
        * SELECT
        * PAN_ZOOM
        * ADD_BOUNDING_BOX

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.

        Raises
        ------
        ValueError
            Raise error if event.mode is not ADD_BOUNDING_BOX, PAN_ZOOM, or SELECT.
        """
        mode_buttons = {
            Mode.SELECT: self.select_button,
            Mode.PAN_ZOOM: self.panzoom_button,
            Mode.ADD_BOUNDING_BOX: self.bounding_box_button,
        }

        if event.mode in mode_buttons:
            mode_buttons[event.mode].setChecked(True)
        else:
            raise ValueError(
                trans._("Mode '{mode}'not recognized", mode=event.mode)
            )

    def changeFaceColor(self, color: np.ndarray):
        """Change face color of bounding boxes.

        Parameters
        ----------
        color : np.ndarray
            Face color for bounding boxes, color name or hex string.
            Eg: 'white', 'red', 'blue', '#00ff00', etc.
        """
        with self.layer.events.current_face_color.blocker():
            self.layer.current_face_color = color

    def changeEdgeColor(self, color: np.ndarray):
        """Change edge color of bounding boxes.

        Parameters
        ----------
        color : np.ndarray
            Edge color for bounding boxes, color name or hex string.
            Eg: 'white', 'red', 'blue', '#00ff00', etc.
        """
        with self.layer.events.current_edge_color.blocker():
            self.layer.current_edge_color = color

    def changeTextColor(self, color: np.ndarray):
        """Change edge color of bounding boxes.

        Parameters
        ----------
        color : np.ndarray
            Edge color for bounding boxes, color name or hex string.
            Eg: 'white', 'red', 'blue', '#00ff00', etc.
        """
        with self.layer.text.events.color.blocker():
            self.layer.text.color = color
            self.layer.refresh()

    def changeWidth(self, value):
        """Change edge line width of bounding boxes on the layer model.

        Parameters
        ----------
        value : float
            Line width of bounding boxes.
        """
        self.layer.current_edge_width = float(value) / 2

    def changeTextSize(self, value):
        """Change edge line width of bounding boxes on the layer model.

        Parameters
        ----------
        value : float
            Line width of bounding boxes.
        """
        self.layer.text.size = float(value) / 2

    def changeSizeMode(self, value=None):
        self.layer.size_mode = value

    def changeSizeMultiplier(self, value):
        self.layer.size_multiplier = value

    def changeSizeConst(self, value):
        self.layer.size_constant = value

    def change_text_visibility(self, state):
        """Toggle the visibiltiy of the text.

        Parameters
        ----------
        state : QCheckBox
            Checkbox indicating if text is visible.
        """
        if state == Qt.Checked:
            self.layer.text.visible = True
        else:
            self.layer.text.visible = False

    def _on_text_visibility_change(self, event):
        """Receive layer model text visibiltiy change change event and update checkbox.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        with self.layer.text.events.visible.blocker():
            self.textDispCheckBox.setChecked(self.layer.text.visible)

    def _on_edge_width_change(self, event=None):
        """Receive layer model edge line width change event and update slider.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        with self.layer.events.edge_width.blocker():
            value = self.layer.current_edge_width
            value = np.clip(int(2 * value), 0, 40)
            self.widthSlider.setValue(value)

    def _on_current_edge_color_change(self, event=None):
        """Receive layer model edge color change event and update color swatch.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        with qt_signals_blocked(self.edgeColorEdit):
            self.edgeColorEdit.setColor(self.layer.current_edge_color)

    def _on_current_face_color_change(self, event=None):
        """Receive layer model face color change event and update color swatch.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        with qt_signals_blocked(self.faceColorEdit):
            self.faceColorEdit.setColor(self.layer.current_face_color)

    def _on_current_text_color_change(self, event=None):
        """Receive layer model face color change event and update color swatch.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        with qt_signals_blocked(self.textColorEdit):
            self.textColorEdit.setColor(self.layer.text.color)

    def _on_editable_change(self, event=None):
        """Receive layer model editable change event & enable/disable buttons.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        disable_with_opacity(
            self,
            [
                'select_button',
                'bounding_box_button',
                'delete_button',
            ],
            self.layer.editable,
        )

    def _on_size_mode_change(self, event=None):
        size_mode = self.layer.size_mode
        self.bb_size_mode_combobox.setCurrentText(size_mode)
        if size_mode == "average":
            self.bb_size_const_label.setVisible(False)
            self.bb_size_const_slider.setVisible(False)
            self.bb_size_mult_label.setVisible(True)
            self.bb_size_mult_slider.setVisible(True)
        elif size_mode == "constant":
            self.bb_size_const_label.setVisible(True)
            self.bb_size_const_slider.setVisible(True)
            self.bb_size_mult_label.setVisible(False)
            self.bb_size_mult_slider.setVisible(False)

    def _on_size_multiplier_change(self, event=None):
        with self.layer.events.size_multiplier.blocker():
            self.bb_size_mult_slider.setValue(self.layer.size_multiplier)

    def _on_size_constant_change(self, event=None):
        with self.layer.events.size_multiplier.blocker():
            self.bb_size_const_slider.setValue(self.layer.size_constant)

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()

from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls
def register_layer_control(layer_type):
    layer_to_controls[layer_type] = QtBoundingBoxControls
