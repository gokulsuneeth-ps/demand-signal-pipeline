"""Tests for wape/rmse/mape: correctness against hand-worked numbers, and
every degenerate-input path raising instead of silently returning nan/0.
"""

from __future__ import annotations

import numpy as np
import pytest
from dsp.models.metrics import mape, rmse, wape


def test_wape_matches_hand_calculation():
    actuals = np.array([10, 20, 30])
    predictions = np.array([12, 18, 33])
    # |2| + |2| + |3| = 7, sum(|actuals|) = 60
    assert wape(actuals, predictions) == pytest.approx(7 / 60)


def test_wape_is_pooled_not_averaged_per_group():
    """The whole point of wape being a plain function over arrays: calling
    it once on concatenated data is pooling. This test exists to make
    the pooling-vs-averaging distinction concrete, matching the Fold
    A/Fold B worked example from the day 4 explainer.
    """
    fold_a_actual, fold_a_pred = np.array([100.0]), np.array([90.0])  # 10% error
    fold_b_actual, fold_b_pred = np.array([1.0]), np.array([0.0])  # 100% error

    averaged = (wape(fold_a_actual, fold_a_pred) + wape(fold_b_actual, fold_b_pred)) / 2
    pooled = wape(
        np.concatenate([fold_a_actual, fold_b_actual]),
        np.concatenate([fold_a_pred, fold_b_pred]),
    )
    assert averaged == pytest.approx(0.55)
    assert pooled == pytest.approx(11 / 101)
    assert pooled < averaged  # the low-volume fold no longer dominates


def test_rmse_matches_hand_calculation():
    actuals = np.array([0.0, 0.0])
    predictions = np.array([3.0, 4.0])
    # sqrt(mean(9, 16)) = sqrt(12.5)
    assert rmse(actuals, predictions) == pytest.approx(np.sqrt(12.5))


def test_mape_excludes_zero_actual_rows():
    actuals = np.array([0, 10, 0, 20])
    predictions = np.array([1, 11, 2, 18])
    # only rows 1 and 3 count: |1/10| = .1, |2/20| = .1 -> mean = .1
    assert mape(actuals, predictions) == pytest.approx(0.1)


@pytest.mark.parametrize("metric", [wape, rmse, mape])
def test_raises_on_length_mismatch(metric):
    with pytest.raises(ValueError):
        metric(np.array([1, 2]), np.array([1, 2, 3]))


@pytest.mark.parametrize("metric", [wape, rmse, mape])
def test_raises_on_empty_input(metric):
    with pytest.raises(ValueError):
        metric(np.array([]), np.array([]))


def test_wape_raises_on_all_zero_actuals():
    with pytest.raises(ValueError):
        wape(np.array([0, 0, 0]), np.array([1, 2, 3]))


def test_mape_raises_on_all_zero_actuals():
    with pytest.raises(ValueError):
        mape(np.array([0, 0]), np.array([1, 2]))


def test_rmse_does_not_raise_on_all_zero_actuals():
    # rmse has no divide-by-zero shape, unlike wape/mape - all-zero
    # actuals are a perfectly valid input for it.
    assert rmse(np.array([0, 0]), np.array([1, 2])) == pytest.approx(np.sqrt(2.5))
