import numpy as np
import pandas as pd
import pytest
from scipy import stats

from recsys.evaluation.compare import compare, holm, paired_p_value


def test_holm_multiplies_by_the_number_of_remaining_tests_and_never_decreases() -> None:
    # Sorted: 0.01 * 3, 0.03 * 2, 0.04 * 1 -> 0.03, 0.06, 0.04; the last is raised to 0.06.
    adjusted = holm(np.array([0.01, 0.04, 0.03]))
    assert adjusted.tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert holm(np.array([0.5, 0.9])).tolist() == [1.0, 1.0]


def test_paired_p_value_matches_scipy_and_handles_identical_models() -> None:
    difference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert paired_p_value(difference) == pytest.approx(stats.ttest_1samp(difference, 0.0).pvalue)
    assert paired_p_value(-difference) == pytest.approx(paired_p_value(difference))
    assert paired_p_value(np.zeros(5)) == 1.0


def test_compare_ranks_models_and_tests_every_pair() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.05, size=500)
    values = pd.DataFrame(
        {
            "weak": 0.2 + noise,
            "strong": 0.3 + noise + rng.normal(0.0, 0.01, size=500),
            "twin": 0.2 + noise,  # scores exactly like "weak" on every user
        }
    )
    out = compare(values, np.random.default_rng(1), n_resamples=200, confidence=0.95, alpha=0.05)

    assert out["n_users"] == 500
    assert out["ranking"] == ["strong", "weak", "twin"]  # equal means keep the column order
    assert list(out["means"]) == out["ranking"]
    pairs = {(pair["better"], pair["worse"]): pair for pair in out["pairs"]}
    assert list(pairs) == [("strong", "weak"), ("strong", "twin"), ("weak", "twin")]

    lead = pairs[("strong", "weak")]
    assert lead["difference"] == pytest.approx(0.1, abs=0.005)
    assert lead["ci_low"] < lead["difference"] < lead["ci_high"]
    assert lead["significant"] and lead["p_adjusted"] >= lead["p_value"]

    tie = pairs[("weak", "twin")]
    assert (tie["difference"], tie["ci_low"], tie["ci_high"]) == (0.0, 0.0, 0.0)
    assert tie["p_value"] == 1.0 and not tie["significant"]


def test_compare_is_reproducible_for_a_seed() -> None:
    values = pd.DataFrame(np.random.default_rng(0).random((50, 3)), columns=["a", "b", "c"])
    first = compare(values, np.random.default_rng(7), 100, 0.95, 0.05)
    second = compare(values, np.random.default_rng(7), 100, 0.95, 0.05)
    assert first == second
