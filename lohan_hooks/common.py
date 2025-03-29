from enum import Enum, auto


class TrainingState(Enum):
    FORWARD = auto()
    BACKWARD = auto()
    IDLE = auto()


class StorageStatus(Enum):
    # not initialized
    UNINITIALIZED = auto()

    # on GPU
    READY = auto()

    # not on GPU
    OFFLOADED = auto()

    OFFLOADING = auto()

    FETCHING = auto()

    # on CPU
    ON_CPU = auto()

    # fetching from cpu
    FETCHING_FROM_CPU = auto()

    # offloading to nvme
    OFFLOADING_TO_CPU = auto()

    # on NVMe
    ON_NVME = auto()

    # fetching from nvme
    FETCHING_FROM_NVME = auto()

    # offloading to nvme
    OFFLOADING_TO_NVME = auto()
