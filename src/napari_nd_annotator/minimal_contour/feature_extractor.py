import napari
import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_filter
from skimage.filters import sobel, sobel_h, sobel_v
from scipy.signal import convolve2d
import cv2

from qtpy.QtCore import QRunnable, QThreadPool, Signal, QObject, Slot
import queue
import threading
import time
import itertools


class FeatureExtractor:
    def __init__(self, max_threads=None):
        self.queueLock = threading.Lock()
        self.queue = queue.Queue()
        self.n_done = 0
        self.pool = QThreadPool.globalInstance()
        self.n_threads = self.pool.maxThreadCount() if max_threads is None else max_threads
        self.done_mask = None

    def start_jobs(self, img, outs, current_slice, dims_not_displayed=None, rgb=None, f=None):
        self.start = time.time()
        ndim = img.ndim - (1 if rgb else 0)
        current_slice = tuple(map(lambda s: 0 if type(s) == slice else s, current_slice))
        idx_list = np.asarray(list(itertools.product(*[[-1] if i not in dims_not_displayed else range(img.shape[i]) for i in range(ndim)])))
        order = np.argsort(np.abs(idx_list-current_slice).sum(1))
        idx_list = idx_list[order]
        idx_list = list(map(lambda l: tuple(l[i] if i in dims_not_displayed else slice(None) for i in range(len(l))), idx_list))
        # viewer.dims.events.current_step.connect(on_current_step)
        self.init_runnables()
        self.done_mask = np.zeros([img.shape[i] for i in dims_not_displayed], bool)
        # Fill the queue
        self.queueLock.acquire()
        for idx in idx_list:
            self.queue.put(idx)
        self.queueLock.release()

        # Create new threads
        for runnable in self.runnables:
            runnable.data = img
            runnable.outs = outs
            runnable.rgb = rgb
            runnable.done_mask = self.done_mask
            runnable.dims_not_displayed = dims_not_displayed
            self.pool.start(runnable)

    def init_runnables(self):
        self.runnables = []
        self.n_done = 0
        for i in range(self.n_threads):
            runnable = self.FeatureExtractorTask(i, self.queue, self.queueLock)
            self.runnables.append(runnable)

    class FeatureExtractorTask(QRunnable):
        def __init__(self, threadID, queue, lock):
            super().__init__()
            self.threadID = threadID
            self.q = queue
            self.lock = lock
            self.data = None
            self.outs = None
            self.rgb = None
            self.done_mask = None
            self.dims_not_displayed = None
            self._signals = self.Signals()
            self.conv_filter_v = np.asarray([[0, -1, 1]]).astype(float)
            self.conv_filter_h = np.asarray([[0], [-1], [1]]).astype(float)

        @Slot()
        def run(self):
            if self.data is None or self.outs is None:
                return
            while True:
                self.lock.acquire()
                if not self.q.empty():
                    idx = self.q.get()
                    self.lock.release()
                    if not self.rgb:
                        # self.out[idx, ...] = gaussian_gradient_magnitude(self.data[idx].astype(float), 5)
                        self.outs[0][idx] = sobel_v(self.data[idx].astype(float))
                        self.outs[1][idx] = sobel_h(self.data[idx].astype(float))
                    else:
                        r, g, b = self.data[idx + (0,)].astype(float), self.data[idx + (1,)].astype(float), self.data[idx + (2,)].astype(float)
                        channels_v = []
                        channels_h = []
                        if np.any(r):
                            channels_v.append(gaussian_filter(r, 2, (0, 1)))
                            channels_h.append(gaussian_filter(r, 2, (1, 0)))
                        if np.any(g):
                            channels_v.append(gaussian_filter(g, 2, (0, 1)))
                            channels_h.append(gaussian_filter(g, 2, (1, 0)))
                        if np.any(b):
                            channels_v.append(gaussian_filter(b, 2, (0, 1)))
                            channels_h.append(gaussian_filter(b, 2, (1, 0)))
                        self.outs[0][idx] = sum(channels_v)
                        self.outs[1][idx] = sum(channels_h)

                    self.q.task_done()
                    self.slice_done.emit(idx)
                    self.done_mask[tuple(idx[i] for i in self.dims_not_displayed)] = True
                else:
                    self.lock.release()
                    break
            self.done.emit()

        @property
        def done(self):
            return self._signals.done

        @property
        def slice_done(self):
            return self._signals.slice_done

        class Signals(QObject):
            done = Signal()
            slice_done = Signal(object)
