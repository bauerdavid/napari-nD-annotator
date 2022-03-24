import math

from cv2 import cv2
import numpy as np
from PyQt5.QtWidgets import QPushButton
from magicgui.widgets import FunctionGui
from napari.layers import Labels, Shapes
from scipy.ndimage import binary_fill_holes
from skimage.draw import draw
import matplotlib.pyplot as plt


class ContourWidget(FunctionGui):
    def __init__(self, viewer):
        super().__init__(
            self.call_fn,
            auto_call=True,
            param_options={
                "out_mode": {
                    "choices": [("Append", "append"), ("Overwrite", "overwrite")]
                },
                "autofill_objects": {
                    "text": "autofill objects"
                }
            }

        )
        self.prev_layer = None
        contour_button = QPushButton("Mask ➝ contour")
        contour_button.clicked.connect(self.create_contours)
        self.native.layout().addWidget(contour_button)
        viewer.dims.events.ndisplay.connect(lambda _: contour_button.setEnabled(viewer.dims.ndisplay == 2))

        contour_to_mask_btn = QPushButton("Contour ➝ mask")
        contour_to_mask_btn.clicked.connect(self.create_masks)
        self.native.layout().addWidget(contour_to_mask_btn)
        self.viewer = viewer

    @property
    def mask_layer(self):
        return self.labels_layer.value

    @property
    def contour_layer(self):
        return self.shapes_layer.value

    def call_fn(self, labels_layer: Labels, shapes_layer: Shapes, autofill_objects=False, out_mode="append"):
        if labels_layer is None:
            return
        if self.prev_layer is not None\
                and self.prev_layer != labels_layer and self.fill_holes in self.prev_layer.mouse_drag_callbacks:
            self.prev_layer.mouse_drag_callbacks.remove(self.fill_holes)
        self.prev_layer = labels_layer
        if autofill_objects and self.fill_holes not in labels_layer.mouse_drag_callbacks:
            labels_layer.mouse_drag_callbacks.append(self.fill_holes)
        elif not autofill_objects and self.fill_holes in labels_layer.mouse_drag_callbacks:
            labels_layer.mouse_drag_callbacks.remove(self.fill_holes)

    def fill_holes(self, layer, event):
        if layer.mode != "paint":
            return
        coordinates = layer.world_to_data(event.position)
        coordinates = tuple(max(0, min(layer.data.shape[i] - 1, int(round(coord)))) for i, coord in enumerate(coordinates))
        image_coords = tuple(coordinates[i] for i in range(len(coordinates)) if i in self.viewer.dims.displayed)
        slice_dims = tuple(coordinates[i] if i in self.viewer.dims.not_displayed else slice(None) for i in range(len(coordinates)))
        current_draw = np.zeros_like(layer.data[slice_dims], np.bool)
        start_x, start_y = prev_x, prev_y = image_coords
        cx, cy = draw.disk((start_x, start_y), layer.brush_size/2)
        cx = np.clip(cx, 0, current_draw.shape[0] - 1)
        cy = np.clip(cy, 0, current_draw.shape[1] - 1)
        current_draw[cx, cy] = True
        yield
        while event.type == 'mouse_move':

            coordinates = layer.world_to_data(event.position)
            coordinates = tuple(max(0, min(layer.data.shape[i] - 1, int(round(coord)))) for i, coord in enumerate(coordinates))
            image_coords = tuple(coordinates[i] for i in range(len(coordinates)) if i in self.viewer.dims.displayed)
            self.draw_line(prev_x, prev_y, image_coords[-2], image_coords[-1], layer.brush_size, current_draw)
            prev_x, prev_y = image_coords
            yield
        # s = np.asarray([[0, 1, 0],
        #                 [1, 1, 1],
        #                 [0, 1, 0]])
        s = None
        coordinates = layer.world_to_data(event.position)
        coordinates = tuple(
            max(0, min(layer.data.shape[i] - 1, int(round(coord)))) for i, coord in enumerate(coordinates))
        image_coords = tuple(coordinates[i] for i in range(len(coordinates)) if i in self.viewer.dims.displayed)
        prev_x, prev_y = image_coords
        self.draw_line(prev_x, prev_y, start_x, start_y, layer.brush_size, current_draw)
        cx, cy = draw.disk((prev_x, prev_y), layer.brush_size/2)
        cx = np.clip(cx, 0, current_draw.shape[0] - 1)
        cy = np.clip(cy, 0, current_draw.shape[1] - 1)
        current_draw[cx, cy] = True
        binary_fill_holes(current_draw, output=current_draw, structure=s)
        layer.data[slice_dims][current_draw] = layer.selected_label
        layer.refresh()

    def draw_line(self, x1, y1, x2, y2, brush_size, output):
        line_x, line_y = draw.line(x1, y1, x2, y2)
        for x, y in zip(line_x, line_y):
            cx, cy = draw.disk((x, y), math.ceil(brush_size/2+0.1))
            cx = np.clip(cx, 0, output.shape[0] - 1)
            cy = np.clip(cy, 0, output.shape[1] - 1)
            output[cx, cy] = True

    def create_contours(self, *args, **kwargs):
        data = self.mask_layer.data
        other_idx = np.meshgrid(*(range(data.shape[i]) if i not in self.viewer.dims.displayed else 0 for i in range(data.ndim)))
        other_idx = list(map(lambda x: x.reshape(-1), other_idx))
        label_values = np.unique(data)[1:]
        contours = []
        contour_idx = []
        for val in label_values:
            mask = data == val
            mask = mask.astype(np.uint8) * 255
            for idx in zip(*other_idx):
                slice_idx = tuple(idx[i] if i not in self.viewer.dims.displayed else slice(None) for i in range(self.mask_layer.ndim))
                if mask[slice_idx].max() == 0:
                    continue
                res = cv2.findContours(mask[slice_idx], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if res[1] is not None:
                    cnts = res[0]
                    cnts = list(filter(lambda cnt: cv2.contourArea(cnt) > 0, cnts))
                    if self.viewer.dims.displayed[0] > self.viewer.dims.displayed[1]:
                        cnts = [np.flip(cnt, -1) for cnt in cnts]
                    contours.extend(cnts)
                    contour_idx.extend([idx]*len(cnts))
        translation = np.asarray(self.mask_layer.translate) - np.asarray(self.contour_layer.translate)
        translation = translation.reshape(1, -1)
        shapes_data = []
        for contour, c_idx in zip(contours, contour_idx):
            contour = np.fliplr(np.squeeze(contour))
            contour_coords = np.tile(np.asarray(c_idx)[np.newaxis, :], (len(contour), 1))
            contour_coords[:, list(self.viewer.dims.displayed)] = contour
            contour_coords += translation
            shapes_data.append(contour_coords)
        if self.out_mode.value == "overwrite":
            self.contour_layer.data = []
        elif self.out_mode.value != "append":
            raise ValueError("out mode must be one of 'append' or 'overwrite'")
        for shape in shapes_data:
            self.contour_layer.add(shape, shape_type="polygon")
        self.contour_layer.refresh()

    def create_masks(self):
        if self.out_mode.value == "append":
            start_index = self.mask_layer.data.max()+1
        elif self.out_mode.value == "overwrite":
            self.mask_layer.data[:] = 0
            start_index = 1
        else:
            raise ValueError("out mode must be one of 'append' or 'overwrite'")
        for label, shape in enumerate(self.contour_layer.data, start=start_index):
            spatial_dims = np.argwhere(~np.all(shape == shape[:1], axis=0)).reshape(-1)
            contour = np.round(shape[:, spatial_dims]).astype(np.int)
            contour = contour - np.asarray([self.mask_layer.translate[i] for i in range(self.mask_layer.ndim) if i in spatial_dims]).reshape(1, -1)
            contour = np.round(np.fliplr(contour))
            contour = contour[:, None, :]
            slice_idx = tuple(slice(None) if i in spatial_dims
                              else int(shape[0, i] - self.mask_layer.translate[i] + self.contour_layer.translate[i])
                              for i in range(shape.shape[1]))
            mask = self.mask_layer.data[slice_idx].astype(np.uint8)
            cv2.drawContours(mask, [contour], 0, label, -1)
            self.mask_layer.data[slice_idx] = mask
            self.mask_layer.refresh()

    def objectName(self):
        return "contour widget"
