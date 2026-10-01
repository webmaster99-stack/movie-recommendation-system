import random

import numpy as np

from recsys.utils.seed import make_deterministic


def test_same_seed_gives_identical_random_streams() -> None:
    def draw() -> tuple[float, float, np.ndarray]:
        rng = make_deterministic(seed=123, num_threads=1)
        return random.random(), float(np.random.rand()), rng.random(5)  # noqa: NPY002

    a, b = draw(), draw()
    assert a[0] == b[0]
    assert a[1] == b[1]
    np.testing.assert_array_equal(a[2], b[2])


def test_different_seeds_differ() -> None:
    x = make_deterministic(seed=1, num_threads=1).random(5)
    y = make_deterministic(seed=2, num_threads=1).random(5)
    assert not np.array_equal(x, y)
