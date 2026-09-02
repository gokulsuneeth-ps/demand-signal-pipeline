"""Rolling-origin cross-validation harness.

Deliberately never a random train/test split - that leaks future
information into training for time series and silently inflates reported
accuracy. Folds are defined globally by calendar date (every series shares
the same train_end / forecast window per fold), not per-series - so that
pooling errors across folds (see metrics.py) is pooling errors from the
same points in time for every series, not an average across series that
happen to be looking at different dates and calling it one number. This
also matches how the pipeline would actually run in production: every SKU
is forecast as of the same "today," not a different one per SKU.

Three responsibilities kept separate on purpose:

- `generate_folds` is pure date arithmetic - no DataFrame, no per-series
  logic. It answers "what are the fold boundaries," and only that, so it's
  testable with a five-line synthetic date range.
- `assemble_fold_frames` does the actual per-series filtering into
  train/test rows for one fold, including the min-history exclusion rule.
  Kept separate so a bug in "which rows belong to fold 2" can never be
  confused with a bug in "where are fold 2's boundaries."
- `run_backtest` wires a single forecaster through every fold and reports
  results; `compares_favorably` takes two already-computed results (a
  candidate model's and seasonal-naive's) and checks PROBLEM_STATEMENT.md's
  actual bar - beats the baseline on EVERY fold, not just on average. Kept
  separate from `run_backtest` itself so "did we run a backtest" and "is
  this model good enough to ship" stay independently testable, and so
  `run_backtest` never has to secretly know it's being compared to anything.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from dsp.models.metrics import mape, rmse, wape

logger = logging.getLogger(__name__)

ForecastFn = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class Fold:
    """One rolling-origin fold's date boundaries.

    A dataclass rather than a bare tuple so calling code reads
    `fold.train_end` instead of `fold[0]` - misordering positional tuple
    elements is a class of bug this makes structurally impossible.
    """

    fold_number: int  # 1-indexed, fold 1 = most recent (last) window
    train_end: pd.Timestamp  # last date included in training (inclusive)
    forecast_start: pd.Timestamp  # first date of the forecast window
    forecast_end: pd.Timestamp  # last date of the forecast window (inclusive)


def generate_folds(dates: pd.Series, n_folds: int, horizon: int) -> list[Fold]:
    """Builds `n_folds` rolling-origin folds packed back-to-back against
    the most recent date in `dates`.

    Fold 1 forecasts the last `horizon` days in the data. Fold 2 forecasts
    the `horizon` days immediately before fold 1's window. And so on -
    folds never overlap, and every fold's training data stops exactly
    where the previous (older) fold's forecast window would have started,
    so nothing in a fold's "future" is ever visible to its own training.

    Boundaries are derived from the actual max date in `dates`, not
    hardcoded - so this is correct for whatever date range the caller
    passes in (the full CA/FOODS subset, a CA_1-only fallback, or a tiny
    synthetic test range) without the function needing to change.

    Raises if `dates` doesn't contain enough distinct days to fit
    `n_folds * horizon` days plus at least one day of training history.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    unique_dates = pd.Series(pd.to_datetime(dates)).drop_duplicates().sort_values()
    if unique_dates.empty:
        raise ValueError("dates must not be empty")

    last_date = unique_dates.iloc[-1]
    first_date = unique_dates.iloc[0]

    required_days = n_folds * horizon + 1  # +1: at least one day to train on
    available_days = (last_date - first_date).days + 1
    if available_days < required_days:
        raise ValueError(
            f"not enough history to build {n_folds} folds of horizon {horizon}: "
            f"need at least {required_days} calendar days, have {available_days}"
        )

    folds = []
    for n in range(1, n_folds + 1):
        forecast_end = last_date - pd.Timedelta(days=(n - 1) * horizon)
        forecast_start = forecast_end - pd.Timedelta(days=horizon - 1)
        train_end = forecast_start - pd.Timedelta(days=1)
        folds.append(
            Fold(
                fold_number=n,
                train_end=train_end,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
            )
        )
    return folds


def assemble_fold_frames(
    df: pd.DataFrame,
    fold: Fold,
    min_train_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Splits `df` into (train_df, test_df) for one fold, per series.

    A series is included only if it has at least `min_train_days` of rows
    on or before `fold.train_end` - a series that started selling too
    recently to have that much history is excluded from THIS FOLD
    specifically (it may well qualify for a later, less-demanding fold),
    rather than silently kept with a too-short or partially-null training
    window. Returns the excluded-series count alongside the frames so the
    caller can log it instead of it disappearing unnoticed.

    `df` must have `id` and `date` columns; `date` is coerced to
    datetime for comparison, matching `generate_folds`.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    train_candidates = out[out["date"] <= fold.train_end]
    test_candidates = out[(out["date"] >= fold.forecast_start) & (out["date"] <= fold.forecast_end)]

    history_counts = train_candidates.groupby("id")["date"].count()
    eligible_ids = history_counts[history_counts >= min_train_days].index

    all_ids = out["id"].unique()
    n_excluded = len(all_ids) - len(eligible_ids)

    train_df = train_candidates[train_candidates["id"].isin(eligible_ids)]
    test_df = test_candidates[test_candidates["id"].isin(eligible_ids)]

    return train_df, test_df, n_excluded


@dataclass(frozen=True)
class FoldResult:
    """One fold's outcome: its boundaries, row counts, and metrics."""

    fold: Fold
    wape: float
    rmse: float
    mape: float
    n_series: int
    n_excluded: int


@dataclass(frozen=True)
class BacktestResult:
    """Full backtest outcome: every fold's result plus the pooled,
    across-all-folds metrics that are the actual headline numbers (see
    metrics.py's docstring on why pooling, not averaging per-fold
    percentages, is the correct way to combine folds).

    `per_fold` is a plain list, not reduced to a single number anywhere
    in this object - PROBLEM_STATEMENT.md's "beats the baseline on every
    fold" requirement depends on the per-fold detail staying visible, not
    just the pooled summary.
    """

    per_fold: list[FoldResult] = field(default_factory=list)
    pooled_wape: float = 0.0
    pooled_rmse: float = 0.0
    pooled_mape: float = 0.0


def run_backtest(
    df: pd.DataFrame,
    forecast_fn: ForecastFn,
    n_folds: int,
    horizon: int,
    min_train_days: int,
) -> BacktestResult:
    """Runs `forecast_fn` through `n_folds` rolling-origin folds and
    reports per-fold and pooled metrics.

    Folds are iterated newest-first (fold 1 = the most recent window),
    matching `generate_folds`' own numbering, so a fold number means the
    same thing everywhere it's reported or logged.

    `forecast_fn` must be `(train_df, test_df) -> predictions_df` with
    `id`, `date`, `prediction` columns - exactly `seasonal_naive_forecast`'s
    shape. A forecaster with a different return shape (e.g. `ets_forecast`,
    which also returns a fallback count) needs a small adapter at the call
    site; this harness deliberately stays ignorant of any forecaster's
    internal diagnostics beyond the predictions themselves.

    Raises if any fold has zero eligible series after `min_train_days`
    filtering - a silently-empty fold would otherwise report an undefined
    WAPE for that fold rather than surfacing the configuration problem.
    """
    dates = df["date"]
    folds = generate_folds(dates, n_folds=n_folds, horizon=horizon)

    per_fold: list[FoldResult] = []
    all_actuals: list[pd.Series] = []
    all_predictions: list[pd.Series] = []

    for fold in folds:
        train_df, test_df, n_excluded = assemble_fold_frames(df, fold, min_train_days)

        n_series = test_df["id"].nunique()
        if n_series == 0:
            raise ValueError(
                f"fold {fold.fold_number} ({fold.forecast_start.date()} to "
                f"{fold.forecast_end.date()}) has zero eligible series after "
                f"min_train_days={min_train_days} filtering ({n_excluded} excluded) - "
                f"lower min_train_days or check the input data's date range"
            )

        predictions_df = forecast_fn(train_df, test_df)
        merged = test_df[["id", "date", "sales"]].merge(
            predictions_df[["id", "date", "prediction"]], on=["id", "date"], how="left"
        )
        if merged["prediction"].isna().any():
            n_missing = merged["prediction"].isna().sum()
            raise ValueError(
                f"fold {fold.fold_number}: forecast_fn returned no prediction for "
                f"{n_missing} test rows - a forecaster must cover every row it was given"
            )

        fold_wape = wape(merged["sales"].to_numpy(), merged["prediction"].to_numpy())
        fold_rmse = rmse(merged["sales"].to_numpy(), merged["prediction"].to_numpy())
        fold_mape = mape(merged["sales"].to_numpy(), merged["prediction"].to_numpy())

        logger.info(
            "fold %d (%s to %s): wape=%.4f n_series=%d n_excluded=%d",
            fold.fold_number,
            fold.forecast_start.date(),
            fold.forecast_end.date(),
            fold_wape,
            n_series,
            n_excluded,
        )

        per_fold.append(
            FoldResult(
                fold=fold,
                wape=fold_wape,
                rmse=fold_rmse,
                mape=fold_mape,
                n_series=n_series,
                n_excluded=n_excluded,
            )
        )
        all_actuals.append(merged["sales"])
        all_predictions.append(merged["prediction"])

    pooled_actuals = pd.concat(all_actuals).to_numpy()
    pooled_predictions = pd.concat(all_predictions).to_numpy()

    return BacktestResult(
        per_fold=per_fold,
        pooled_wape=wape(pooled_actuals, pooled_predictions),
        pooled_rmse=rmse(pooled_actuals, pooled_predictions),
        pooled_mape=mape(pooled_actuals, pooled_predictions),
    )


@dataclass(frozen=True)
class ComparisonResult:
    """Fold-by-fold comparison of a candidate model against a baseline.

    `passed_every_fold` is the literal PROBLEM_STATEMENT.md bar - True
    only if the candidate's WAPE is strictly lower than the baseline's on
    every single fold, not just on the pooled average. `fold_deltas` keeps
    the per-fold detail (baseline_wape - candidate_wape, positive means
    the candidate is better) so a failing comparison shows exactly which
    fold(s) fell short instead of just a pass/fail bit.
    """

    passed_every_fold: bool
    fold_deltas: list[float]
    pooled_delta: float


def compares_favorably(
    candidate_result: BacktestResult, baseline_result: BacktestResult
) -> ComparisonResult:
    """Compares two already-computed BacktestResults fold-by-fold.

    Takes results, not models or forecast functions - `run_backtest` is
    called twice at the call site (once per model), and this function's
    only job is the comparison itself, which keeps it trivially testable
    with hand-built BacktestResult fixtures rather than needing real
    forecasters or data.

    Raises if the two results don't have the same number of folds, or if
    their fold numbers don't line up 1:1 - comparing fold 2 of one result
    against fold 3 of another would silently produce a meaningless
    comparison.
    """
    if len(candidate_result.per_fold) != len(baseline_result.per_fold):
        raise ValueError(
            f"cannot compare results with different fold counts: "
            f"candidate has {len(candidate_result.per_fold)}, "
            f"baseline has {len(baseline_result.per_fold)}"
        )

    fold_deltas = []
    for cand_fold, base_fold in zip(
        candidate_result.per_fold, baseline_result.per_fold, strict=True
    ):
        if cand_fold.fold.fold_number != base_fold.fold.fold_number:
            raise ValueError(
                f"fold number mismatch: candidate fold {cand_fold.fold.fold_number} "
                f"vs baseline fold {base_fold.fold.fold_number} - results must be "
                f"generated from the same fold configuration"
            )
        fold_deltas.append(base_fold.wape - cand_fold.wape)

    return ComparisonResult(
        passed_every_fold=all(d > 0 for d in fold_deltas),
        fold_deltas=fold_deltas,
        pooled_delta=baseline_result.pooled_wape - candidate_result.pooled_wape,
    )
