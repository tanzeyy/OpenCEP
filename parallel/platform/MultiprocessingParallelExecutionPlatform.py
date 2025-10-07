"""
Provides parallelization functionality based on Python multiprocess library.
"""

from typing import Any, Callable, Optional

import multiprocess
from parallel.platform.ParallelExecutionPlatform import Lock
from parallel.platform.ParallelExecutionPlatform import ParallelExecutionPlatform
from parallel.platform.ParallelExecutionPlatform import ParallelExecutionUnit


class MultiprocessingParallelExecutionPlatform(ParallelExecutionPlatform):
    """
    Creates execution unit objects based on Python multiprocess.
    """

    @staticmethod
    def create_parallel_execution_unit(unit_id: int, callback_function: Callable, *args, **kwargs):
        """
        - `daemon`: if not None, set process.daemon explicitly.
        - `use_queue_ipc`: when True, create a pair of queues for send/receive.
        - NOTE: Ensure `callback_function`, `args`, `kwargs` are picklable on spawn platforms.
        """
        # Optionally pass unit_id to the worker (common need)
        # kwargs.setdefault("_unit_id", unit_id)

        new_proc = multiprocess.Process(target=callback_function, args=args, kwargs=kwargs, name=f"MPUnit-{unit_id}")

        return MultiprocessingParallelExecutionUnit(unit_id=unit_id, process=new_proc)

    @staticmethod
    def create_lock():
        return MultiprocessingLock()


class MultiprocessingParallelExecutionUnit(ParallelExecutionUnit):
    """
    A parallel execution unit wrapping a single Python process.
    """

    def __init__(
        self,
        unit_id: int,
        process: multiprocess.Process,
    ):
        super().__init__(unit_id)
        self._process = process

    def start(self) -> None:
        """Start the underlying process."""
        self._process.start()

    def stop(self) -> None:
        return self._process.terminate()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until the process exits or timeout occurs.
        Returns True if the process exited before timeout, else False.
        """
        self._process.join(timeout)
        return not self._process.is_alive()

    def send(self, data: Any) -> None:
        """
        Send data to the worker via input queue.
        """
        return

    def receive(self, timeout: Optional[float] = None) -> Any:
        """
        Receive data from the worker via output queue.
        Raises queue.Empty on timeout.
        """
        return

    # Optional helpers
    def is_alive(self) -> bool:
        """Return whether the process is alive."""
        return self._process.is_alive()

    def exitcode(self) -> Optional[int]:
        """Return exit code if finished, else None."""
        return self._process.exitcode


class MultiprocessingLock(Lock):
    def __init__(self):
        self._lock = multiprocess.Lock()

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        # `timeout` must be None or >= 0 when blocking=True.
        if blocking and timeout is not None and timeout < 0:
            raise ValueError("timeout must be None or >= 0")
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()
