import os
import tempfile
import atexit
import warnings

import numpy as np
import random
import string
import shutil
from .feature_extractor import FeatureExtractor
from napari import Viewer
from typing import Union, Optional
import glob
TEMP_SUFFIX = "_nd_annotator"


class FeatureManager:
    def __init__(self, viewer):
        self.layer = None
        self.dims_displayed = None
        self.memmaps: list[Optional[Union[np.ndarray, np.memmap]]] = [None, None]
        self.slices_calculated = dict()
        self.clean_tmp()
        self.temp_folder = tempfile.mkdtemp(suffix=TEMP_SUFFIX)
        # map layers to file prefix
        self.prefix_map = dict()
        self.viewer: Viewer = viewer
        self.feature_extractor = FeatureExtractor()
        atexit.register(self.clean)

    def get_features(self, layer, block=True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dims_displayed = tuple(layer._dims_displayed)
            dims_not_displayed = tuple(layer._dims_not_displayed)
        if layer != self.layer or dims_displayed != self.dims_displayed:
            self.init_file(layer, dims_displayed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx = tuple(layer._slice_indices[i] for i in range(layer.ndim) if i in dims_not_displayed)
        # if not block and not self.slices_calculated[layer][dims_displayed][idx]:
        if not block and not self.feature_extractor.done_mask[idx]:
            raise ValueError("features not calculated for layer %s at %s" % (layer, idx))
        # while not self.slices_calculated[layer][dims_displayed][idx]:
        while not self.feature_extractor.done_mask[idx]:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx = tuple(layer._slice_indices[i] if i in dims_not_displayed else slice(None) for i in range(layer.ndim))
        return self.memmaps[0][idx], self.memmaps[1][idx]

    def init_file(self, layer, dims_displayed):
        self.layer = layer
        self.dims_displayed = dims_displayed
        while len(self.memmaps) > 0:
            # TODO check if it is safe to close
            del self.memmaps[0]
        if layer in self.prefix_map:
            filename = self.prefix_map[layer]
        else:
            filename = self.random_prefix()
            self.prefix_map[layer] = filename
        if layer not in self.slices_calculated:
            self.slices_calculated[layer] = dict()
        if dims_displayed not in self.slices_calculated[layer]:
            self.slices_calculated[layer][dims_displayed] = np.zeros([layer.data.shape[i] for i in self.viewer.dims.not_displayed], bool)
        path_v = self.generate_filename(filename, dims_displayed, "_v")
        path_h = self.generate_filename(filename, dims_displayed, "_h")
        if layer.rgb:
            shape = layer.data.shape[:-1]
        else:
            shape = layer.data.shape
        if not os.path.exists(path_v):
            self.memmaps.append(np.memmap(path_v, shape=shape, dtype=float, mode="w+"))
            self.memmaps.append(np.memmap(path_h, shape=shape, dtype=float, mode="w+"))
            self.start_feature_calculation(layer)
        else:
            self.memmaps.append(np.memmap(path_v, shape=shape, dtype=float))
            self.memmaps.append(np.memmap(path_h, shape=shape, dtype=float))
            self.feature_extractor.done_mask = np.ones([shape[i] for i in layer._dims_not_displayed], bool)

    def generate_filename(self, prefix, dims_displayed, suffix=''):
        return os.path.join(self.temp_folder, "tmp_ftrs_%s_%s%s.dat" % (prefix, "_".join(str(d) for d in dims_displayed), suffix))

    def start_feature_calculation(self, layer):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slice_indices = layer._slice_indices
            dims_displayed = layer._dims_displayed
        self.feature_extractor.start_jobs(layer.data, self.memmaps, slice_indices, dims_displayed, layer.rgb)

    @staticmethod
    def random_prefix():
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))

    def clean(self):
        try:
            del self.memmaps[-1]
        except:
            pass
        try:
            del self.memmaps[-1]
        except:
            pass
        if os.path.exists(self.temp_folder):
            shutil.rmtree(self.temp_folder)

    def clean_tmp(self):
        temp_dir = tempfile.gettempdir()
        temp_folders = glob.glob(os.path.join(temp_dir, "%s*%s" % (tempfile.gettempprefix(), TEMP_SUFFIX)))
        for fold in temp_folders:
            shutil.rmtree(fold)
