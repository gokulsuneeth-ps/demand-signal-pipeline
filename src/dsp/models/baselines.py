"""Seasonal-naive and ETS baselines.

These are not throwaway comparisons - PROBLEM_STATEMENT.md defines "done"
as beating seasonal-naive on every backtest fold, so this module's output
is the bar the real model has to clear, and it ships in the same
CI-tested path as the LightGBM model.

Both `seasonal_naive_forecast` and `ets_forecast` take the same
(train_df, test_df) shape that `backtest.assemble_fold_frames` produces,
and both return a DataFrame with `id`, `date`, `prediction` - same shape
as test_df's identifying columns, so lining predictions up against
actuals is a plain merge regardless of which baseline produced them.
"""

from __future__ import annotations

import logging
import warnings

import pandas as pd

logger = logging.getLogger(__name__)


def seasonal_naive_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    season_length: int = 7,
) -> pd.DataFrame:
    """Predicts each (id, date) in test_df by cycling that series' own
    actual sales from the LAST REAL WEEK of history in train_df.

    For a horizon no wider than one season (the common case), this is
    just "use the value from `season_length` days earlier," which is a
    real actual day in train_df. For a wider horizon (e.g. this
    project's real 28-day horizon with season_length=7), day N days past
    train_end can't simply look back N-7 days for N > 7 - that day is
    itself inside the forecast window, not an actual. Instead this
    repeats the last real season (the final `season_length` days of
    train_df) as many times as needed: the lookback for a day `offset`
    days past train_end is `offset` days minus however many whole
    seasons have elapsed, i.e.

        lookback_date = date - season_length * ceil(offset / season_length)

    which always lands on a real actual date in train_df (never on
    another forecasted day), and is a strict generalization of the
    previous `date - season_length` rule: whenever horizon <=
    season_length, ceil(offset / season_length) == 1 for every day in
    the window, so this reduces to exactly the old lookup. This is also
    the same convention the M5 competition itself uses for its own
    28-day seasonal-naive baseline.

    Also raises if a specific computed lookup is missing from train_df.
    Given `assemble_fold_frames` already guarantees `min_train_days` of
    history per included series, this should never trigger for a
    correctly configured harness - if it does, min_train_days is set too
    small relative to season_length, and that is a configuration bug
    worth surfacing immediately rather than as a mysterious WAPE number
    three layers away.
    """
    train = train_df.copy()
    train["date"] = pd.to_datetime(train["date"])
    train_end = train["date"].max()
    lookup = train.set_index(["id", "date"])["sales"]

    test = test_df.copy()
    test["date"] = pd.to_datetime(test["date"])
    offset_days = (test["date"] - train_end).dt.days
    # offset_days > 0 is the normal case (test strictly follows train_end):
    # ceil(offset/season_length) seasons back. Some callers (e.g.
    # ets_forecast's per-series fallback) pass a train_df/test_df pair
    # whose "train" wasn't cut at a hard fold boundary and can overlap the
    # test window; for those (offset_days <= 0) fall back to exactly one
    # season back, matching this function's original, unconditional
    # `date - season_length` behavior rather than inventing a new rule for
    # a case this fold-based cycling logic isn't meant to describe.
    ceil_seasons = -(-offset_days // season_length)  # ceil division, valid for offset_days > 0
    seasons_elapsed = ceil_seasons.where(offset_days > 0, 1)
    lookback_dates = test["date"] - pd.to_timedelta(season_length * seasons_elapsed, unit="D")
    lookup_keys = list(zip(test["id"], lookback_dates, strict=True))

    missing = [k for k in lookup_keys if k not in lookup.index]
    if missing:
        raise ValueError(
            f"seasonal_naive_forecast: {len(missing)} (id, date) lookups missing "
            f"from train_df, e.g. {missing[:3]} - min_train_days is likely too "
            f"small relative to season_length"
        )

    predictions = [lookup[k] for k in lookup_keys]
    nan_at = [k for k, p in zip(lookup_keys, predictions, strict=True) if pd.isna(p)]
    if nan_at:
        raise ValueError(
            f"seasonal_naive_forecast: {len(nan_at)} lookups resolved to a NaN "
            f"sales value in train_df (present but null), e.g. {nan_at[:3]} - "
            f"this is a data quality problem, not a missing-history problem, and "
            f"must not be silently forecast as NaN"
        )
    return pd.DataFrame(
        {"id": test["id"].to_numpy(), "date": test["date"].to_numpy(), "prediction": predictions}
    )


_ets_import_checked = False
_ets_import_error: Exception | None = None


def _ets_dependencies_available() -> tuple[bool, Exception | None]:
    """Checks once (and caches) whether statsforecast's AutoETS can
    actually be imported in this environment.

    This exists because of a real bug caught on a Windows machine with an
    Application Control policy that blocks scipy's compiled `cython_blas`
    DLL: statsforecast (and therefore scipy) failed to import with a raw
    ImportError. The first version of `ets_forecast` imported
    statsforecast BEFORE its own try/except block, so that ImportError
    propagated straight past the batch -> per-series -> naive fallback
    chain entirely uncaught - the fallback design only protected against
    failures happening INSIDE the try, not an environment that can't load
    the library at all. Checking (and caching) availability up front,
    before any fold-processing loop, means a broken environment fails
    over to seasonal-naive for the whole run once, loudly logged, instead
    of re-attempting (and re-failing) a DLL load once per series across
    thousands of series.
    """
    global _ets_import_checked, _ets_import_error
    if not _ets_import_checked:
        try:
            from statsforecast import StatsForecast  # noqa: F401
            from statsforecast.models import AutoETS  # noqa: F401
        except ImportError as e:
            _ets_import_error = e
        _ets_import_checked = True
    return _ets_import_error is None, _ets_import_error


def _assert_shared_forecast_window(test_df: pd.DataFrame, n_series: int) -> None:
    """Raises unless every series in test_df has the exact same set of
    forecast dates.

    `ets_forecast` computes a single `horizon` from test_df's overall
    min/max date and passes it to statsforecast as one number for the
    whole batch - correct only when every series shares the same
    forecast window, which `assemble_fold_frames` guarantees in normal
    use (fold windows are global-by-date; see backtest.py). Nothing
    enforced that assumption here, and violating it produced a silently
    wrong horizon rather than a clear error at the point of the actual
    problem - caught during testing with a fixture that (unrealistically)
    gave two series different forecast windows. Checking it explicitly
    turns a confusing downstream error into an immediate, accurate one.
    """
    windows = test_df.groupby("id")["date"].apply(
        lambda s: tuple(sorted(pd.to_datetime(s).tolist()))
    )
    unique_windows = windows.unique()
    if len(unique_windows) > 1:
        raise ValueError(
            f"ets_forecast requires every series to share the same forecast "
            f"date window, but found {len(unique_windows)} distinct windows "
            f"across {n_series} series - this test_df did not come from a "
            f"single assemble_fold_frames() fold"
        )


def ets_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    season_length: int = 7,
) -> tuple[pd.DataFrame, int]:
    """Per-series AutoETS forecast, `horizon` steps ahead per series,
    where horizon is inferred from test_df's date range.

    If statsforecast's dependencies aren't usable in this environment at
    all (see `_ets_dependencies_available`), falls back to seasonal-naive
    for every series in this fold immediately, with n_fallback equal to
    the full series count - this is an environment problem, not a
    per-series data problem, so it's handled as one fallback for the
    whole fold rather than thousands of repeated failed attempts.

    Otherwise, fits across all eligible series in one batched
    statsforecast call first (fast path - this is what makes ETS
    tractable at thousands-of-series scale instead of a slow Python
    loop). If that batched call raises - which a single degenerate series
    (near-constant, extreme values, too little effective variation) can
    trigger for the ENTIRE batch, not just its own row - falls back to
    fitting series one at a time for this fold only, catching failures
    per series and using seasonal-naive for any series ETS can't fit on
    its own.

    Returns (predictions_df, n_fallback) where n_fallback is the count of
    series that needed the seasonal-naive fallback - always returned
    alongside the predictions so a caller can log it, never silently
    absorbed.
    """
    n_series = test_df["id"].nunique()

    available, import_error = _ets_dependencies_available()
    if not available:
        logger.warning(
            "ETS is unavailable in this environment (%s) - falling back to "
            "seasonal-naive for all %d series in this fold",
            import_error,
            n_series,
        )
        predictions = seasonal_naive_forecast(train_df, test_df, season_length)
        return predictions, n_series

    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS

    _assert_shared_forecast_window(test_df, n_series)

    test_dates = pd.to_datetime(test_df["date"])
    horizon = (test_dates.max() - test_dates.min()).days + 1

    train = train_df.copy()
    train["date"] = pd.to_datetime(train["date"])
    sf_input = train.rename(columns={"id": "unique_id", "date": "ds", "sales": "y"})[
        ["unique_id", "ds", "y"]
    ]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sf = StatsForecast(models=[AutoETS(season_length=season_length)], freq="D", n_jobs=1)
            fcst = sf.forecast(df=sf_input, h=horizon)
        predictions = _fcst_to_predictions(fcst, test_df, "AutoETS")
        return predictions, 0
    except Exception:
        # Batch call failed - isolate per series so one bad series can't
        # take the whole fold's ETS forecast down with it.
        return _ets_forecast_per_series(sf_input, train_df, test_df, season_length, horizon)


def _ets_forecast_per_series(
    sf_input: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    season_length: int,
    horizon: int,
) -> tuple[pd.DataFrame, int]:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS

    all_predictions = []
    n_fallback = 0

    for series_id, series_df in sf_input.groupby("unique_id"):
        series_test = test_df[test_df["id"] == series_id]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sf = StatsForecast(
                    models=[AutoETS(season_length=season_length)], freq="D", n_jobs=1
                )
                fcst = sf.forecast(df=series_df, h=horizon)
            preds = _fcst_to_predictions(fcst, series_test, "AutoETS")
        except Exception:
            n_fallback += 1
            series_train = train_df[train_df["id"] == series_id]
            preds = seasonal_naive_forecast(series_train, series_test, season_length)
        all_predictions.append(preds)

    return pd.concat(all_predictions, ignore_index=True), n_fallback


def _fcst_to_predictions(fcst: pd.DataFrame, test_df: pd.DataFrame, model_col: str) -> pd.DataFrame:
    """Reshapes statsforecast's (unique_id, ds, <model_col>) output back
    to this module's (id, date, prediction) convention, and asserts the
    result covers exactly the rows test_df asked for - not more, not
    fewer - so a silent partial-coverage mismatch can't slip through.
    """
    out = fcst.rename(columns={"unique_id": "id", "ds": "date", model_col: "prediction"})
    out["date"] = pd.to_datetime(out["date"])

    test = test_df.copy()
    test["date"] = pd.to_datetime(test["date"])
    merged = test[["id", "date"]].merge(out, on=["id", "date"], how="left")

    if merged["prediction"].isna().any():
        missing = merged[merged["prediction"].isna()][["id", "date"]]
        raise ValueError(
            f"ETS forecast missing predictions for {len(missing)} test rows, "
            f"e.g. {missing.head(3).to_dict('records')}"
        )

    return merged
