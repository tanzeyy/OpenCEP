from enum import Enum


class ParallelExecutionPlatforms(Enum):
    """
    Supported platforms for parallel and/or distributed execution.
    """
    THREADING = 0
    MULTIPROCESSING = 1

    # TODO: should support more types
