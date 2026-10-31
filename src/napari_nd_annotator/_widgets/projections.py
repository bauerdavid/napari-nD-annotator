import itertools

import numpy as np
from PIL import Image as PILImage
from qtpy.QtCore import QSettings, Qt, QTimer
from qtpy.QtGui import QImage, QPixmap, QResizeEvent, QColor
from qtpy.QtWidgets import QLabel, QSizePolicy, QWidget, QVBoxLayout, QSlider, QHBoxLayout, QGridLayout, QCheckBox, \
    QColorDialog, QPushButton, QWIDGETSIZE_MAX, QDockWidget
from napari.layers.labels._labels_constants import Mode
from qtpy import QtCore

from .._helper_functions import layer_dims_not_displayed, layer_slice_indices
from matplotlib import colors

class DataBox:
    """A part of the data of a layer, given by its inclusive corners in data (index) coordinates.

    The projections of an object only show its bounding box. They work on numpy views of the layer data, so nothing is
    copied, and their slice indices are relative to the first corner of the box.
    """

    def __init__(self, box_min, box_max):
        self.set(box_min, box_max)

    def set(self, box_min, box_max):
        self.min = np.maximum(np.asarray(box_min, dtype=int), 0)
        self.max = np.maximum(np.asarray(box_max, dtype=int), self.min)

    @property
    def idx(self):
        return tuple(slice(int(lo), int(hi) + 1) for lo, hi in zip(self.min, self.max))


def _box_min(box, ndim):
    return np.zeros(ndim, dtype=int) if box is None else np.asarray(box.min, dtype=int)


class DataProjectionWidget(QLabel):
    def __init__(self, viewer, image_layer, mask_layer, displayed_axes, slices=None,
                 flip_image=False, crosshair_color=None, box=None):
        super().__init__()
        self._crosshair_color = None
        self.box = box
        self.viewer = viewer
        self.image_layer = image_layer
        self.mask_layer = mask_layer
        self.flip_image = flip_image
        self.max_width, self.max_height = QWIDGETSIZE_MAX, QWIDGETSIZE_MAX
        self.slices = list(slices) or [s//2 for s in self.image_data.shape]
        self.displayed_axes = displayed_axes
        self._overlay = np.zeros(self.overlay_shape, np.uint8)
        if crosshair_color is None:
            self.crosshair_color = (255, 0, 0, 160)
        else:
            self.crosshair_color = crosshair_color
        self.update()
        self.pixmap = None
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.mousePressEvent = self.on_click
        self.setFrameStyle(QLabel.Raised | QLabel.Box)

    def setSlices(self, *args, update=True):
        assert len(args) % 2 == 0
        for i in range(0, len(args), 2):
            self.slices[args[i]] = max(min(args[i+1], self.image_data.shape[args[i]]-1), 0)
        if update:
            self.update()

    @property
    def overlay_shape(self):
        # the image can be replaced by one of a different size (e.g. when the bounding box of the object changes)
        return tuple(self.image_data.shape[i] for i in range(self.image_layer.ndim) if self.im_idx[i] == slice(None))[:2]

    def sync_with_data(self):
        """Keeps the slices and the overlay valid for the current shape of the image."""
        for dim in range(min(len(self.slices), self.image_layer.ndim)):
            self.slices[dim] = max(min(self.slices[dim], self.image_data.shape[dim]-1), 0)
        if self._overlay.shape != self.overlay_shape:
            self._overlay = np.zeros(self.overlay_shape, np.uint8)

    @property
    def crosshair_color(self):
        return self._crosshair_color

    @crosshair_color.setter
    def crosshair_color(self, new_color):
        if all(map(lambda v: 0.<=v<=1., new_color)):
            new_color = map(lambda v: int(v*255), new_color)
        self._crosshair_color = tuple(new_color)
        self.update()

    @property
    def overlay(self):
        overlay = (np.ones(self._overlay.shape + (4,)) * self.crosshair_color).astype(np.uint8)
        overlay[..., -1] *= self._overlay
        if self.flip_image:
            overlay = np.transpose(overlay, (1, 0, 2))
        return PILImage.fromarray(overlay)

    @overlay.setter
    def overlay(self, new_overlay):
        self._overlay = new_overlay > 0

    def update_overlay(self, layer=None, coordinates=None):
        self.sync_with_data()
        overlay = np.zeros(self.overlay_shape, bool)
        if layer is not None and coordinates is not None:
            # the overlay axes follow the order of the dimensions, the displayed axes are reversed if the image is flipped
            row_axis, col_axis = reversed(self.displayed_axes) if self.flip_image else self.displayed_axes
            row, col = coordinates[row_axis], coordinates[col_axis]
            if all(layer_dim == proj_dim for layer_dim, proj_dim in
                   zip(sorted(self.viewer.dims.displayed), sorted(self.displayed_axes))):
                if 0 <= row < overlay.shape[0]:
                    overlay[row] = True
                if 0 <= col < overlay.shape[1]:
                    overlay[:, col] = True
            else:
                # the brush extent, or a single pixel for the tools without a brush
                half_size = layer.brush_size // 2 if layer.mode in (Mode.ERASE, Mode.PAINT) else 0
                for dim in self.viewer.dims.displayed:
                    if dim == row_axis and 0 <= col < overlay.shape[1]:
                        overlay[max(row - half_size, 0):max(row + half_size + 1, 0), col] = True
                    elif dim == col_axis and 0 <= row < overlay.shape[0]:
                        overlay[row, max(col - half_size, 0):max(col + half_size + 1, 0)] = True
        elif (layer is None) != (coordinates is None):
            raise ValueError("layer and coordinates should be both provided or left None")
        self.overlay = overlay

    @property
    def image_data(self):
        data = self.image_layer.data
        return data if self.box is None else data[self.box.idx]

    @property
    def mask_data(self):
        data = self.mask_layer.data
        return data if self.box is None else data[self.box.idx]

    @property
    def im_idx(self):
        return tuple(
                self.slices[i] if i not in self.displayed_axes
                    else slice(None) for i in range(self.image_layer.ndim))

    def update(self, update_icon=True, new_size=None):
        if update_icon:
            self.sync_with_data()
            if self.image_layer.rgb:
                im = self.image_data[self.im_idx]
            else:
                im = self.image_data[self.im_idx]
                max_ = self.image_layer.contrast_limits[1]
                min_ = self.image_layer.contrast_limits[0]
                im = np.clip((im-min_)/(max_-min_), 0, 1)
                im = (self.image_layer.colormap.map(im.ravel()).reshape(im.shape + (4,))*255).astype(np.uint8)[..., :3]

            if self.mask_data.shape[:self.image_layer.ndim] == self.image_data.shape[:self.image_layer.ndim]:
                mask = (self.mask_data[self.im_idx] > 0).astype(np.uint8)
            else:
                # the mask and the image are not resized at the same time
                mask = np.zeros(im.shape[:2], np.uint8)
            alpha = mask * 180
            mask = np.tile(mask[..., np.newaxis], (1, 1, 4))*255

            mask[..., -1] = alpha

            if self.flip_image:
                im = np.transpose(im, (1, 0, 2))
                mask = np.transpose(mask, (1, 0, 2))
            im = PILImage.fromarray(im.astype(np.uint8)).convert(mode="RGBA")
            mask = PILImage.fromarray(mask).convert(mode="RGBA")
            icon = PILImage.alpha_composite(im, mask)
            icon = PILImage.alpha_composite(icon, self.overlay).convert("RGB")
            icon = np.asarray(icon)
            _icon = icon.reshape(icon.shape[0], icon.shape[1], -1)
            if _icon.shape[-1] == 1:
                _icon = np.tile(_icon, (1, 1, 3))
            elif _icon.shape[-1] == 4:
                _icon = _icon[..., :-1]
            self.img = QImage(_icon, _icon.shape[1], _icon.shape[0], _icon.shape[1]*3, QImage.Format.Format_RGB888)
        pixmap = QPixmap()
        pixmap = pixmap.fromImage(self.img, QtCore.Qt.ImageConversionFlag.ColorOnly)
        self.setMinimumSize(64, 64)
        if new_size is not None:
            w = new_size.width()
            h = new_size.height()
        else:
            w = self.size().width()
            h = self.size().height()
        pixmap = pixmap.scaled(w, h, QtCore.Qt.KeepAspectRatio)
        self.setPixmap(pixmap)

    def on_resize(self, new_size):
        self.update(update_icon=False, new_size=new_size)

    def resizeEvent(self, a0: QResizeEvent) -> None:
        self.on_resize(a0.size())

    def on_click(self, _):
        order = tuple(filter(lambda x: x not in self.displayed_axes, self.viewer.dims.order))
        order = order + tuple(self.displayed_axes)
        self.viewer.dims.order = order
        # the slices are data indices of the layer, the viewer position is in world coordinates
        point = self.image_layer.data_to_world(np.asarray(self.slices) + _box_min(self.box, self.image_layer.ndim))
        other_dims = [dim for dim in range(len(self.viewer.dims.current_step)) if dim not in self.displayed_axes]
        self.viewer.dims.set_point(other_dims, [point[dim] for dim in other_dims])


class SliceDisplayWidget(QWidget):
    # the projections are rendered at most this often (in ms) while the labels or the cursor keep changing
    REFRESH_INTERVAL = 30

    def __init__(self, viewer, image_layer, mask_layer, channels_dim=None, *args, box=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.box = box
        self.setObjectName("Projections")
        main_layout = QVBoxLayout()
        self.settings = QSettings()
        self.overlay_color = self.settings.value("projection_overlay_color", (255, 0, 0, 160))
        self.slider_labels = []
        self.sliders = []
        self.slider_widgets = []
        self.viewer = viewer
        self.image_layer = image_layer
        self.mask_layer = mask_layer
        self.shown = False
        self._connections = []
        self._detached = False
        self._cursor_position = None
        self._cursor_moved = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(self.REFRESH_INTERVAL)
        self._refresh_timer.timeout.connect(self.refresh)
        for dim in range(viewer.dims.ndim):
            label = QLabel("%d" % dim)
            slider = QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(max(self.data_shape[dim] - 1, 0) if dim < len(self.data_shape) else 0)
            slider.setSingleStep(1)
            slider.setFixedWidth(100)
            slider.valueChanged.connect(self.slider_callback(dim))
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(label)
            slider_layout.addWidget(slider)
            slider_widget = QWidget()
            slider_widget.setLayout(slider_layout)
            main_layout.addWidget(slider_widget)
            self.slider_widgets.append(slider_widget)
            if dim not in viewer.dims.order[-viewer.dims.ndisplay:]:
                slider_widget.setVisible(False)
            self.sliders.append(slider)
        grid_layout = QGridLayout()
        self.projections = []
        for dim_pair in itertools.combinations(range(image_layer.ndim), 2):
            if channels_dim in dim_pair:
                continue
            if dim_pair[-2] == viewer.dims.order[-1] or dim_pair[-1] == viewer.dims.order[-2]:
                dim_pair = tuple(reversed(dim_pair))
                flip = True
            else:
                flip = False
            slice_indices = layer_slice_indices(image_layer)
            box_min = _box_min(box, image_layer.ndim)
            slices = [max(min(int(slice_indices[dim]) - int(box_min[dim]), self.data_shape[dim]-1), 0) if dim in layer_dims_not_displayed(image_layer) else 0 for dim in range(viewer.dims.ndim)]
            projection = DataProjectionWidget(viewer, image_layer, mask_layer, dim_pair, slices=slices, flip_image=flip, crosshair_color=self.overlay_color, box=box)
            grid_layout.setRowStretch(len(self.projections)//3, 1)
            grid_layout.setColumnStretch(len(self.projections)%3, 1)
            grid_layout.addWidget(projection, len(self.projections)//3, len(self.projections)%3, QtCore.Qt.AlignmentFlag.AlignCenter)
            self.projections.append(projection)
        self.grid_widget = QWidget()
        self.grid_widget.setLayout(grid_layout)
        self.grid_widget.setSizePolicy(self.grid_widget.sizePolicy().horizontalPolicy(), QSizePolicy.Expanding)
        main_layout.addWidget(self.grid_widget)
        extra_settings_layout = QHBoxLayout()
        self.dockable_checkbox = QCheckBox("dockable")
        dockable = self.settings.value("projectionsWidgetDockable", 'false')
        self.dockable_checkbox.setChecked(dockable == "true")
        self.dockable_checkbox.clicked.connect(self.set_dockable)
        extra_settings_layout.addWidget(self.dockable_checkbox)
        self.change_color_button = QPushButton()
        self.change_color_button.setFlat(True)
        self.change_color_button.setStyleSheet("background-color: %s;" % colors.to_hex(list(map(lambda v: v/255, self.overlay_color))))
        self.change_color_button.clicked.connect(self.show_overlay_colorpicker)
        self.change_color_button.setFixedWidth(20)
        self.change_color_button.setFixedHeight(20)
        self.setToolTip("Change overlay color")

        extra_settings_layout.addWidget(self.change_color_button)
        main_layout.addLayout(extra_settings_layout)
        self.setLayout(main_layout)
        self.setSizePolicy(self.grid_widget.sizePolicy().horizontalPolicy(), QSizePolicy.Expanding)
        self._connect_events()

    @property
    def data_shape(self):
        """Shape of the data the projections show (the box of the layer, if there is one)."""
        shape = self.image_layer.data.shape
        if self.box is None:
            return tuple(shape)
        return tuple(len(range(*s.indices(n))) for s, n in zip(self.box.idx, shape))

    def set_box(self, box_min, box_max):
        """Shows another part of the data (the bounding box of the object changed)."""
        self.box.set(box_min, box_max)
        self.on_step_change()
        self.update_slider_ranges()

    def _connect(self, emitter, callback):
        emitter.connect(callback)
        self._connections.append((emitter, callback))

    def _connect_events(self):
        if self._connections or self._detached:
            return
        self._connect(self.viewer.dims.events.current_step, self.on_step_change)
        self._connect(self.viewer.dims.events.order, self.on_order_change)
        # a paint stroke only emits labels_update (napari >= 0.4.19), setting the data or refreshing the layer
        # (e.g. the minimal contour tool, interpolation) emits data and set_data
        for event_name in ("data", "set_data", "labels_update"):
            emitter = getattr(self.mask_layer.events, event_name, None)
            if emitter is not None:
                self._connect(emitter, self._schedule_refresh)
        for event_name in ("data", "contrast_limits", "colormap"):
            self._connect(getattr(self.image_layer.events, event_name), self._schedule_refresh)
        self._connect(self.viewer.cursor.events.position, self._on_cursor_move)
        self._connect(self.viewer.layers.events.removed, self._on_layer_removed)

    def _disconnect_events(self):
        self._refresh_timer.stop()
        for emitter, callback in self._connections:
            try:
                emitter.disconnect(callback)
            except (ValueError, KeyError, RuntimeError):
                pass
        self._connections = []

    def detach(self):
        """Stops following the viewer and the layers for good (the widget is going away)."""
        self._detached = True
        self._disconnect_events()

    def _on_layer_removed(self, event):
        if event.value in (self.image_layer, self.mask_layer):
            self.detach()

    def _schedule_refresh(self, event=None):
        if self._connections and not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def _on_cursor_move(self, event=None):
        if self.viewer.dims.ndisplay == 3:
            return
        self._cursor_position = tuple(self.viewer.cursor.position)
        self._cursor_moved = True
        self._schedule_refresh()

    def refresh(self):
        """Updates the crosshair and renders the projections."""
        self._refresh_timer.stop()
        if self._detached:
            return
        self.update_slider_ranges(refresh=False)
        if self._cursor_position is not None and self.viewer.dims.ndisplay != 3:
            self.show_crosshair(update_sliders=self._cursor_moved)
        self._cursor_moved = False
        for projection in self.projections:
            projection.update()

    def show_crosshair(self, layer=None, event=None, update_sliders=True):
        if self.viewer.dims.ndisplay == 3:
            return
        layer = self.mask_layer if layer is None else layer
        position = self._cursor_position if event is None else event.position
        if position is None:
            return
        float_coordinates = layer.world_to_data(position)
        coordinates = np.round(np.asarray(float_coordinates)).astype(int)
        if len(coordinates) != self.image_layer.ndim:
            coordinates = coordinates[-self.image_layer.ndim:]
        coordinates = coordinates - _box_min(self.box, len(coordinates))
        shape = self.data_shape
        for p in self.projections:
            p.update_overlay(layer, coordinates)
        if not update_sliders:
            return
        for i, slider in enumerate(self.sliders):
            if i in self.viewer.dims.order[-self.viewer.dims.ndisplay:] and shape[i] > coordinates[i] >= 0:
                slider.blockSignals(True)
                slider.setValue(int(coordinates[i]))
                slider.blockSignals(False)
                for p in self.projections:
                    p.setSlices(i, int(coordinates[i]), update=False)

    def on_step_change(self, event=None, layers=None):
        # the viewer position is in world coordinates, the slices are data indices of the layer
        point = np.asarray(self.image_layer.world_to_data(self.viewer.dims.point))
        point = point - _box_min(self.box, len(point))
        for projection in self.projections:
            projection.setSlices(*itertools.chain(*((dim, int(round(point[dim]))) for dim in self.viewer.dims.not_displayed)))

    def update_slider_ranges(self, refresh=True):
        shape = self.data_shape
        for dim in range(len(self.sliders)):
            self.sliders[dim].setMaximum(max(shape[dim] - 1, 0) if dim < len(shape) else 0)
        for projection in self.projections:
            projection.sync_with_data()
        if refresh:
            self._schedule_refresh()

    def slider_callback(self, dim):
        def on_slider_change(val):
            for projection in self.projections:
                projection.setSlices(dim, val)
        return on_slider_change

    def on_order_change(self, _):
        for dim, slider in enumerate(self.slider_widgets):
            slider.setVisible(dim in self.viewer.dims.order[-self.viewer.dims.ndisplay:])
        for p in self.projections:
            if p.displayed_axes[-2] == self.viewer.dims.order[-1] or p.displayed_axes[-1] == self.viewer.dims.order[-2]:
                p.flip_image = not p.flip_image
                p.displayed_axes = type(p.displayed_axes)(reversed(p.displayed_axes))

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(640, 480)

    def show_overlay_colorpicker(self):
        color = QColorDialog.getColor(QColor(*self.overlay_color), options=QColorDialog.ShowAlphaChannel)
        if color.isValid():
            self.overlay_color = color.red(), color.green(), color.blue(), color.alpha()
            self.settings.setValue("projection_overlay_color", self.overlay_color)
            self.change_color_button.setStyleSheet("background-color: %s;" % colors.to_hex(list(map(lambda v: v/255, self.overlay_color))))
            for projection in self.projections:
                projection.crosshair_color = self.overlay_color

    def set_dockable(self, state):
        self.parent().setAllowedAreas(Qt.AllDockWidgetAreas if state else Qt.NoDockWidgetArea)

    def showEvent(self, QShowEvent):
        self._connect_events()
        if self.shown:
            return
        self.shown = True
        dock_widget = self.parent()
        dock_widget.setFeatures(dock_widget.features() & ~QDockWidget.DockWidgetClosable)
        geometry = self.settings.value("projectionsWidgetGeometry", None)
        is_floating = self.settings.value("projectionsWidgetFloating", None)
        if is_floating is not None:
            dock_widget.setFloating(is_floating == 'true')
        if geometry is not None:
            dock_widget.setGeometry(geometry)
        self.set_dockable(self.dockable_checkbox.isChecked())

    def hideEvent(self, QHideEvent):
        self._disconnect_events()
        self.settings.setValue("projectionsWidgetGeometry", self.parent().geometry())
        self.settings.setValue("projectionsWidgetFloating", self.parent().isFloating())
        self.settings.setValue("projectionsWidgetDockable", self.dockable_checkbox.isChecked())
        super().hideEvent(QHideEvent)

    def closeEvent(self, QCloseEvent):
        self._disconnect_events()
        super().closeEvent(QCloseEvent)
