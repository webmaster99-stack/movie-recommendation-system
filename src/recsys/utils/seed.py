"""Determinism helpers: every entry point calls `make_deterministic` before doing any work.

Seeding alone isn't enough. Multithreaded BLAS/OpenMP can sum floats in a different order
run to run, so thread counts are pinned too. Prefer passing the returned Generator around
explicitly over relying on numpy's global random state.
"""

import os
import random

import numpy as np
from threadpoolctl import threadpool_limits

_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def make_deterministic(seed: int, num_threads: int) -> np.random.Generator:
    for var in _THREAD_ENV_VARS:
        os.environ[var] = str(num_threads)  # for libraries/subprocesses not yet loaded
    threadpool_limits(limits=num_threads)  # for BLAS/OpenMP pools already loaded

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - some third-party libraries still use global state
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.set_num_threads(num_threads)
        torch.use_deterministic_algorithms(True)

    return np.random.default_rng(seed)
