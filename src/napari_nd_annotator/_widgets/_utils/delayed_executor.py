from qtpy.QtCore import QMutex, QObject, QThread, Signal
import time


class DelayedQueue:
    def __init__(self):
        self._first = None
        self._second = None

    def enqueue(self, elem):
        if self._first is None:
            self._first = elem
        else:
            self._second = elem

    def get(self):
        return self._first

    def pop(self):
        self._first = self._second
        self._second = None


class DelayedExecutor(QObject):
    processing = Signal("PyQt_PyObject")
    processed = Signal("PyQt_PyObject")

    def __init__(self, func, parent=None):
        super().__init__(parent)
        self._func = func
        self._arg_queue = DelayedQueue()
        self._mutex = QMutex()
        self._worker = self.DelayedWorker(func, self._arg_queue, self._mutex, self)
        self.worker_thread = QThread()
        self._worker.moveToThread(self.worker_thread)
        self.worker_thread.destroyed.connect(self._worker.deleteLater)
        self.worker_thread.started.connect(self._worker.run)
        self.worker_thread.start()
        if parent is not None:
            parent.destroyed.connect(self.worker_thread.deleteLater)

    def __call__(self, *args, **kwargs):
        self._mutex.lock()
        self._arg_queue.enqueue((args, kwargs))
        self._mutex.unlock()

    class DelayedWorker(QObject):
        def __init__(self, func, q: DelayedQueue, mutex: QMutex, executor):
            super().__init__()
            self._func = func
            self._queue = q
            self._mutex = mutex
            self._executor = executor

        def run(self):
            while True:
                time.sleep(0.1)
                if self._mutex.tryLock():
                    args = self._queue.get()
                    self._mutex.unlock()
                    if args is not None:
                        self._executor.processing.emit(args)
                        self._func(*(args[0]), **(args[1]))
                        self._mutex.lock()
                        self._queue.pop()
                        self._mutex.unlock()
                        self._executor.processed.emit(args)
