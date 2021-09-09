import math

import cv2.cv2
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
                }
            }

        )
        self.prev_layer = None
        contour_button = QPushButton("Mask ➝ contour")
        contour_button.clicked.connect(self.create_contours)
        self.native.layout().addWidget(contour_button)

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
        current_draw = np.zeros_like(layer.data[coordinates[:-2]], np.bool)
        start_x, start_y = prev_x, prev_y = coordinates[-2:]
        cx, cy = draw.disk((start_x, start_y), layer.brush_size/2)
        cx = np.clip(cx, 0, current_draw.shape[0] - 1)
        cy = np.clip(cy, 0, current_draw.shape[1] - 1)
        current_draw[cx, cy] = True
        yield
        while event.type == 'mouse_move':

            coordinates = layer.world_to_data(event.position)
            coordinates = tuple(max(0, min(layer.data.shape[i] - 1, int(round(coord)))) for i, coord in enumerate(coordinates))
            self.draw_line(prev_x, prev_y, coordinates[-2], coordinates[-1], layer.brush_size, current_draw)
            prev_x, prev_y = coordinates[-2:]
            yield
        # s = np.asarray([[0, 1, 0],
        #                 [1, 1, 1],
        #                 [0, 1, 0]])
        s = None
        coordinates = layer.world_to_data(event.position)
        coordinates = tuple(
            max(0, min(layer.data.shape[i] - 1, int(round(coord)))) for i, coord in enumerate(coordinates))
        prev_x, prev_y = coordinates[-2:]
        self.draw_line(prev_x, prev_y, start_x, start_y, layer.brush_size, current_draw)
        cx, cy = draw.disk((prev_x, prev_y), layer.brush_size/2)
        cx = np.clip(cx, 0, current_draw.shape[0] - 1)
        cy = np.clip(cy, 0, current_draw.shape[1] - 1)
        current_draw[cx, cy] = True
        binary_fill_holes(current_draw, output=current_draw, structure=s)
        layer.data[coordinates[:-2]][current_draw] = layer.selected_label
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
        orig_shape = data.shape
        # data = np.moveaxis(data, self.layer._dims_displayed, [0, 1])
        other_idx = np.meshgrid(*(range(data.shape[i]) if i not in self.mask_layer._dims_displayed else 0 for i in range(data.ndim)))
        other_idx = list(map(lambda x: x.reshape(-1), other_idx))
        label_values = np.unique(data)[1:]
        contours = []
        contour_idx = []
        for val in label_values:
            mask = data == val
            mask = mask.astype(np.uint8) * 255
            for idx in zip(*other_idx):
                slice_idx = tuple(idx[i] if i not in self.mask_layer._dims_displayed else slice(None) for i in range(self.mask_layer.ndim))
                if mask[slice_idx].max() == 0:
                    continue
                res = cv2.cv2.findContours(mask[slice_idx], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if res[1] is not None:
                    contours.extend(res[0])
                    contour_idx.extend([idx]*len(res[0]))
        translation = np.asarray(self.mask_layer.translate) - np.asarray(self.contour_layer.translate)
        translation = translation.reshape(1, -1)
        shapes_data = []
        for contour, c_idx in zip(contours, contour_idx):
            contour = np.fliplr(np.squeeze(contour))
            contour_coords = np.tile(np.asarray(c_idx)[np.newaxis, :], (len(contour), 1))
            contour_coords[:, list(self.mask_layer._dims_displayed)] = contour
            contour_coords += translation
            shapes_data.append(contour_coords)
        if self.out_mode.value == "append":
            for shape in shapes_data:
                self.contour_layer.add(shape, shape_type="polygon")
        elif self.out_mode.value == "overwrite":
            self.contour_layer.data = shapes_data
        else:
            raise ValueError("out mode must be one of 'append' or 'overwrite'")
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
            contour = np.round(np.fliplr(shape[:, spatial_dims])[:, None, :]).astype(np.int)
            slice_idx = tuple(slice(None) if i in spatial_dims else int(shape[0, i]) for i in range(shape.shape[1]))
            cv2.drawContours(self.mask_layer.data[slice_idx], [contour], 0, label, -1)
            self.mask_layer.refresh()

    def objectName(self):
        return "contour widget"
