import itertools

import numpy as np
from PIL import Image as PILImage
from PyQt5.Qt import QWIDGETSIZE_MAX
from PyQt5.QtGui import QImage, QPixmap, QResizeEvent
from PyQt5.QtWidgets import QLabel, QSizePolicy, QWidget, QVBoxLayout, QSlider, QHBoxLayout, QGridLayout
from napari.layers.labels._labels_constants import Mode
from qtpy import QtCore



class DataProjectionWidget(QLabel):
    def __init__(self, viewer, image_layer, mask_layer, displayed_axes, image_colormap=None, mask_colormap=None, slices=None, flip_image=False):
        super().__init__()
        self.viewer = viewer
        self.image_layer = image_layer
        self.mask_layer = mask_layer
        self.flip_image = flip_image
        self.max_width, self.max_height = QWIDGETSIZE_MAX, QWIDGETSIZE_MAX
        self.slices = list(slices) or [s//2 for s in self.image_data.shape]
        self.displayed_axes = displayed_axes
        self._overlay = np.zeros(self.image_data[self.im_idx].shape[:2], np.uint8)
        self.image_colormap = image_colormap
        self.mask_colormap = mask_colormap
        self.update()
        self.pixmap = None
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.mousePressEvent = self.on_click
        self.setFrameStyle(QLabel.Raised | QLabel.Box)

    def setSlices(self, *args):
        assert len(args) % 2 == 0
        for i in range(0, len(args), 2):
            self.slices[args[i]] = max(min(args[i+1], self.image_data.shape[args[i]]-1), 0)
        self.update()

    @property
    def overlay(self):
        overlay = np.zeros(self._overlay.shape + (4,), np.uint8)
        overlay[..., -1] = self._overlay*160
        if self.flip_image:
            overlay = np.transpose(overlay, (1, 0, 2))
        return PILImage.fromarray(overlay)

    @overlay.setter
    def overlay(self, new_overlay):
        self._overlay = new_overlay > 0

    def update_overlay(self, layer=None, coordinates=None):
        overlay = np.zeros(self.image_data[self.im_idx].shape[:2], bool)
        if layer is not None and coordinates is not None:
            if layer._mode in [Mode.ERASE, Mode.PAINT]:
                if all(layer_dim == proj_dim for layer_dim, proj_dim in
                       zip(sorted(layer._dims_displayed), sorted(self.displayed_axes))):
                    if 0 <= coordinates[self.displayed_axes[0]] < overlay.shape[1 if self.flip_image else 0]:
                        if self.flip_image:
                            overlay[:, coordinates[self.displayed_axes[0]]] = True
                        else:
                            overlay[coordinates[self.displayed_axes[0]]] = True
                    if 0 <= coordinates[self.displayed_axes[1]] < overlay.shape[0 if self.flip_image else 1]:
                        if self.flip_image:
                            overlay[coordinates[self.displayed_axes[1]]] = True
                        else:
                            overlay[:, coordinates[self.displayed_axes[1]]] = True
                else:
                    for dim in layer._dims_displayed:
                        if dim == self.displayed_axes[0]:
                            start = max(coordinates[dim] - layer.brush_size // 2, 0)
                            end = min(max(coordinates[dim] + layer.brush_size // 2 + 1, 0), overlay.shape[1 if self.flip_image else 0])
                            if self.flip_image and coordinates[self.displayed_axes[1]] < overlay.shape[0]:
                                overlay[coordinates[self.displayed_axes[1]], start:end] = True
                            elif coordinates[self.displayed_axes[1]] < overlay.shape[1]:
                                overlay[start:end, coordinates[self.displayed_axes[1]]] = True
                        elif dim == self.displayed_axes[1]:
                            start = max(coordinates[dim] - layer.brush_size // 2, 0)
                            end = min(max(coordinates[dim] + layer.brush_size // 2 + 1, 0), overlay.shape[0 if self.flip_image else 1])
                            if self.flip_image and coordinates[self.displayed_axes[0]] < overlay.shape[1]:
                                overlay[start:end, coordinates[self.displayed_axes[0]]] = True
                            elif coordinates[self.displayed_axes[0]] < overlay.shape[0]:
                                overlay[coordinates[self.displayed_axes[0]], start:end] = True
        elif bool(layer) ^ bool(coordinates): # XOR
            raise ValueError("layer and coordinates should be both provided or left None")
        self.overlay = overlay

    @property
    def image_data(self):
        return self.image_layer.data

    @property
    def mask_data(self):
        return self.mask_layer.data

    @property
    def im_idx(self):
        return tuple(
                self.slices[i] if i not in self.displayed_axes else slice(None) for i in range(self.image_data.ndim))

    def update(self, update_icon=True, new_size=None):
        if update_icon:
            if self.image_colormap is None:
                im = (self.image_data[self.im_idx]//256)
            else:
                im = (self.image_colormap[self.image_data[self.im_idx]//256]*255)
            if self.flip_image:
                im = np.transpose(im, (1, 0, 2))
            im = PILImage.fromarray(im.astype(np.uint8)).convert(mode="RGBA")
            if self.mask_colormap is None:
                mask = (self.mask_data[self.im_idx] > 0).astype(np.uint8)
                alpha = np.ones_like(mask)*127
                mask = np.tile(mask[..., np.newaxis], (1, 1, 3))
                mask = np.append(mask, alpha[..., np.newaxis], -1)
            else:
                mask = (self.mask_colormap[self.mask_data[self.im_idx]]*255).astype(np.uint8)
                mask[..., -1] = (mask.max(2) > 0).astype(np.uint8)*180
            if self.flip_image:
                mask = np.transpose(mask, (1, 0, 2))
            mask = PILImage.fromarray(mask).convert(mode="RGBA")
            icon = PILImage.alpha_composite(im, mask)
            icon = PILImage.alpha_composite(icon, self.overlay).convert("RGB")
            icon = np.asarray(icon)
            _icon = icon.reshape(icon.shape[0], icon.shape[1], -1)
            if _icon.shape[-1] == 1:
                _icon = np.tile(_icon, (1, 1, 3))
            elif _icon.shape[-1] == 4:
                _icon = _icon[..., :-1]
            self.img = QImage(_icon, _icon.shape[1], _icon.shape[0],
                         _icon.shape[1] * 3, QImage.Format.Format_RGB888)
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
        self.viewer.dims.current_step = tuple(self.viewer.dims.current_step[dim] if dim in self.displayed_axes
                                         else self.slices[dim]+self.image_layer.translate[dim]
                                         for dim in range(len(self.viewer.dims.current_step)))


class SliceDisplayWidget(QWidget):
    def __init__(self, viewer, image_layer, mask_layer, channels_dim=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setObjectName("Projections")
        main_layout = QVBoxLayout()
        self.slider_labels = []
        self.sliders = []
        self.slider_widgets = []
        self.offset = image_layer.translate
        self.viewer = viewer
        self.image_layer = image_layer
        for dim in range(viewer.dims.ndim):
            label = QLabel("%d" % dim)
            slider = QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(image_layer.data.shape[dim])
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
        for dim_pair in itertools.combinations(range(image_layer.data.ndim), 2):
            if channels_dim in dim_pair:
                continue
            if dim_pair[-2] == viewer.dims.order[-1] or dim_pair[-1] == viewer.dims.order[-2]:
                dim_pair = tuple(reversed(dim_pair))
                flip = True
            else:
                flip = False
            slices = [int(viewer.dims.current_step[dim]-image_layer.translate[dim]) if dim in image_layer._dims_not_displayed else 0 for dim in range(viewer.dims.ndim)]
            projection = DataProjectionWidget(viewer, image_layer, mask_layer, dim_pair, image_colormap=image_layer.colormap.colors, mask_colormap=mask_layer.colormap.colors, slices=slices, flip_image=flip)
            grid_layout.setRowStretch(len(self.projections)//3, 1)
            grid_layout.setColumnStretch(len(self.projections)%3, 1)
            grid_layout.addWidget(projection, len(self.projections)//3, len(self.projections)%3, QtCore.Qt.AlignmentFlag.AlignCenter)
            self.projections.append(projection)
        self.grid_widget = QWidget()
        self.grid_widget.setLayout(grid_layout)
        self.grid_widget.setSizePolicy(self.grid_widget.sizePolicy().horizontalPolicy(), QSizePolicy.Expanding)
        main_layout.addWidget(self.grid_widget)
        self.setLayout(main_layout)
        self.setSizePolicy(self.grid_widget.sizePolicy().horizontalPolicy(), QSizePolicy.Expanding)
        viewer.dims.events.current_step.connect(self.on_step_change)
        viewer.dims.events.order.connect(self.on_order_change)
        mask_layer.mouse_drag_callbacks.append(self.update_layer)
        mask_layer.mouse_move_callbacks.append(self.show_crosshair)

    def show_crosshair(self, layer, event):
        if self.viewer.dims.ndisplay == 3:
            return

        float_coordinates = layer.world_to_data(event.position)
        coordinates = np.round(np.asarray(float_coordinates)).astype(int)
        for p in self.projections:
            p.update_overlay(layer, coordinates)
        for i, slider in enumerate(self.sliders):
            if i in self.viewer.dims.order[-self.viewer.dims.ndisplay:] and layer.data.shape[i] > coordinates[i] >= 0:
                slider.setValue(coordinates[i])

    def on_step_change(self, event=None, layers=None):
        for projection in self.projections:
            projection.setSlices(*itertools.chain(*((dim, int(event.source.current_step[dim]-self.offset[dim])) for dim in event.source.not_displayed)))

    def update_layer(self, layer, event):
        self.show_crosshair(layer, event)
        for projection in self.projections:
            projection.update()
        yield
        while event.type == "mouse_move":
            self.show_crosshair(layer, event)
            for projection in self.projections:
                projection.update()
            yield
        for projection in self.projections:
            projection.update()

    def update_slider_ranges(self):
        for dim in range(len(self.sliders)):
            self.sliders[dim].setMaximum(self.image_layer.data.shape[dim])

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
