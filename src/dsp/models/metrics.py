"""WAPE / MAPE / RMSE - defined once, imported everywhere they're reported.

Every function here takes plain arrays, not DataFrames: metrics don't need
to know about `id`, `date`, or fold structure. That structure is entirely
backtest.py's responsibility - it decides which rows to pool together
before calling into this module. In particular, "pool errors across all
folds before dividing" (rather than averaging three per-fold percentages)
is a decision backtest.py makes by concatenating arrays before calling
`wape`, not something this module knows or enforces. There is exactly one
`wape` function here, not a `wape_per_fold` and a `wape_pooled` variant -
keeping only one definition is what prevents the harness, MLflow logging,
and the dashboard from silently disagreeing about what "16% error" means.

Every function below raises rather than silently returning `nan` or `0` on
a degenerate input (empty arrays, a zero-sum denominator, mismatched
lengths). A silent `nan` that slips into an aggregate report is a bug that
hides; a raised `ValueError` forces whoever is running the backtest to
look at the actual data question it surfaces (usually: a fold or series
with genuinely zero sales across its whole window).
"""

from __future__ import annotations

import numpy as np


def _validate(actuals: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shared input checks for every metric below.

    Converts to float arrays and asserts shape/length agreement. No
    coercion beyond that - no broadcasting, no assumption about which
    array is "supposed to be" longer.
    """
    actuals = np.asarray(actuals, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    if actuals.ndim != 1 or predictions.ndim != 1:
        raise ValueError(
            f"actuals and predictions must be 1-D, got shapes "
            f"{actuals.shape} and {predictions.shape}"
        )
    if len(actuals) != len(predictions):
        raise ValueError(
            f"actuals and predictions must be the same length, "
            f"got {len(actuals)} and {len(predictions)}"
        )
    if len(actuals) == 0:
        raise ValueError("actuals and predictions must not be empty")

    return actuals, predictions


def wape(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Weighted Absolute Percentage Error, pooled: sum(|error|) / sum(|actual|).

    Computed once across whatever rows are handed in - the caller decides
    whether that's one fold or all folds concatenated. Raises if
    sum(|actuals|) == 0 (every actual is zero): the metric is undefined
    there, and returning 0/nan would silently misreport a data condition
    (a genuinely zero-sales window) as a model result.
    """
    actuals, predictions = _validate(actuals, predictions)

    denominator = np.abs(actuals).sum()
    if denominator == 0:
        raise ValueError(
            "wape is undefined when sum(|actuals|) == 0 "
            "(every actual value is zero over this window)"
        )

    return float(np.abs(actuals - predictions).sum() / denominator)


def rmse(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Root Mean Squared Error, pooled the same way as wape."""
    actuals, predictions = _validate(actuals, predictions)
    return float(np.sqrt(np.mean((actuals - predictions) ** 2)))


def mape(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Mean Absolute Percentage Error, computed only over rows where the
    actual value is nonzero.

    MAPE is undefined wherever actual == 0 (division by zero). Rather than
    letting that produce inf/nan and propagate silently, rows with
    actual == 0 are excluded from the mean - documented here explicitly so
    "what does this number mean" never depends on tracing numpy's default
    div-by-zero behavior. If every actual is zero, there is nothing left
    to average and this raises, same as wape.
    """
    actuals, predictions = _validate(actuals, predictions)

    nonzero = actuals != 0
    if not nonzero.any():
        raise ValueError("mape is undefined when every actual value is zero")

    pct_errors = np.abs((actuals[nonzero] - predictions[nonzero]) / actuals[nonzero])
    return float(pct_errors.mean())
