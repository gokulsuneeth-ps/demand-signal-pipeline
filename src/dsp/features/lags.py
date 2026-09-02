"""Lag and rolling-window features.

Watch for leakage here specifically: any rolling stat must only look
backward from the forecast origin, never forward - this is the single
most common bug in demand-forecasting feature pipelines and the first
thing a reviewer should check for.

Concretely: a naive `df.groupby("id")["sales"].rolling(7).mean()`
computes each day's 7-day average INCLUDING that day's own sales value -
information the model would never actually have at forecast time, since
forecasting today's sales using today's own sales is circular. Every
rolling function here calls `.shift(1)` before `.rolling(...)`, so a
day's rolling stat reflects only the days strictly BEFORE it. This is
verified directly by a dedicated leakage test in tests/test_lags.py, not
just asserted in a comment.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_LAGS = (7, 14, 28)
DEFAULT_ROLLING_WINDOWS = (7, 28)


def add_lag_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    id_col: str = "id",
    date_col: str = "date",
) -> pd.DataFrame:
    """Adds `sales_lag_<N>` for each N in `lags`: that series' own sales
    value exactly N days earlier.

    Requires `df` to already be sorted by (id_col, date_col) with no
    missing calendar days per series - `shift(N)` is a positional shift,
    not a date-aware one, so a gap in a series' date range would
    silently produce a lag value from the wrong actual day. `df` is
    re-sorted defensively inside this function rather than trusting the
    caller, but a gap in dates is NOT checked here (that is
    `assemble_fold_frames`'/backtest-time concern, not this pure
    feature function's) - stated explicitly since it's a real,
    non-obvious precondition.
    """
    out = df.sort_values([id_col, date_col]).copy()
    grouped_sales = out.groupby(id_col, observed=True)["sales"]
    for lag in lags:
        out[f"sales_lag_{lag}"] = grouped_sales.shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    id_col: str = "id",
    date_col: str = "date",
) -> pd.DataFrame:
    """Adds `sales_rollmean_<N>` and `sales_rollstd_<N>` for each N in
    `windows`: the trailing N-day mean/std of that series' sales, NOT
    including the current day (see module docstring - `.shift(1)` before
    `.rolling(N)` is what enforces this).

    `min_periods` is left at pandas' default (equal to the window size),
    so a day without a full N days of prior history gets NaN rather than
    a rolling stat quietly computed from fewer, weaker data points and
    presented as equally reliable - consistent with this project's
    established preference (day 3/5) for leaving missing signal as NaN
    for the model to see, not zero-filling or otherwise disguising it.
    """
    out = df.sort_values([id_col, date_col]).copy()
    shifted = out.groupby(id_col, observed=True)["sales"].shift(1)
    out["_shifted_sales_for_rolling"] = shifted

    for window in windows:
        rolling = out.groupby(id_col, observed=True)["_shifted_sales_for_rolling"].rolling(window)
        out[f"sales_rollmean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
        out[f"sales_rollstd_{window}"] = rolling.std().reset_index(level=0, drop=True)

    out = out.drop(columns=["_shifted_sales_for_rolling"])
    return out
