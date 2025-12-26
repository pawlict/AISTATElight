from __future__ import annotations
from typing import Callable, Any
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

class TaskSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int)
    log = Signal(str)

class BackgroundTask(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.kwargs.setdefault("progress_cb", self.signals.progress.emit)
            self.kwargs.setdefault("log_cb", self.signals.log.emit)
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())

class TaskRunner:
    def __init__(self) -> None:
        self.pool = QThreadPool.globalInstance()

    def submit(self, task: BackgroundTask) -> None:
        self.pool.start(task)