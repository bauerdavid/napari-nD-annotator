import numpy as np
from cv2 import cv2
from magicgui.widgets import FunctionGui
from napari.layers import Labels
from scipy.interpolate import interp1d


def contour_cv2_mask_uniform(mask, contoursize_max):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    max_ind = np.argmax(areas)
    contour = np.squeeze(contours[max_ind])
    contour = np.reshape(contour, (-1, 2))
    contour = np.append(contour, contour[0].reshape((-1, 2)), axis=0)
    contour = contour.astype('float32')

    rows, cols = mask.shape
    delta = np.diff(contour, axis=0)
    s = [0]
    for d in delta:
        dl = s[-1] + np.linalg.norm(d)
        s.append(dl)

    if (s[-1] == 0):
        s[-1] = 1

    s = np.array(s) / s[-1]
    fx = interp1d(s, contour[:, 0] / rows, kind='linear')
    fy = interp1d(s, contour[:, 1] / cols, kind='linear')
    S = np.linspace(0, 1, contoursize_max, endpoint=False)
    X = rows * fx(S)
    Y = cols * fy(S)

    contour = np.transpose(np.stack([X, Y])).astype(np.float32)

    contour = np.stack((contour[:, 1], contour[:, 0]), axis=-1)
    return contour


class InterpolationWidget(FunctionGui):
    def __init__(self, viewer):
        super().__init__(
            self.interpolate,
        )
        self.viewer = viewer
        self.labels_layer.native.currentIndexChanged.connect(self.update_dim_limit)

    def interpolate(self, labels_layer: Labels, dimension: int, n_contour_points=500):
        if labels_layer is None:
            return
        data = labels_layer.data
        layer_slice_template = [
            slice(None) if d in labels_layer._dims_displayed
                else None if d == dimension
                else self.viewer.dims.current_step[d]
            for d in range(labels_layer.ndim)]
        prev_cnt = None
        prev_layer = None
        for i in range(data.shape[dimension]):
            layer_slice = layer_slice_template.copy()
            layer_slice[dimension] = i
            mask = data[tuple(layer_slice)] == labels_layer.selected_label
            mask = mask.astype(np.uint8)
            if mask.max() == 0:
                continue
            cnt = contour_cv2_mask_uniform(mask, n_contour_points)
            centroid = cnt.mean(0)
            start_index = np.argmin(np.arctan2(*(cnt - centroid).T))
            cnt = np.roll(cnt, start_index, 0)
            if prev_cnt is not None:
                for j in range(prev_layer + 1, i):
                    inter_layer_slice = layer_slice_template.copy()
                    inter_layer_slice[dimension] = j
                    prev_w = i - j
                    cur_w = j - prev_layer
                    mean_cnt = (prev_w * prev_cnt + cur_w * cnt)/(prev_w + cur_w)
                    mean_cnt = mean_cnt.astype(np.int32)
                    mask = np.zeros_like(data[tuple(inter_layer_slice)])
                    cv2.drawContours(mask, [np.flip(mean_cnt, -1)], 0, labels_layer.selected_label, -1)
                    data[tuple(inter_layer_slice)] = mask
            prev_cnt = cnt
            prev_layer = i
            labels_layer.refresh()

    def update_dim_limit(self, event):
        if self.labels_layer.value is not None:
            self.dimension.native.setMaximum(self.labels_layer.value.ndim)
        else:
            self.dimension.native.setMaximum(0)
