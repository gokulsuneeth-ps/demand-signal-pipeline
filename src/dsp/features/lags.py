"""Lag and rolling-window features on `sales` — the target variable.

The single rule that matters more than anything else in this file: a
feature for day t must only use information available strictly before
day t. `sales` is the thing being forecast, so any statistic derived from
it has to be shifted before use — get this wrong and the backtest in
day 4 will report a suspiciously good number that has no relationship to
real forecasting accuracy, because the model was secretly allowed to see
the answer.

Contrast with `sell_price` (see prices.py): a retailer sets prices in
advance, so today's price is legitimately known before today's sales
happen. That is NOT the same situation as `sales` itself, and is not
lagged the same way.
"""

from __future__ import annotations

import pandas as pd

LAG_DAYS = [7, 14, 28]
ROLLING_WINDOWS = [7, 28]


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Every function below assumes rows are in (id, date) order — pandas
    `groupby(...).shift()` / `.rolling()` operate on row order, not on the
    `date` column's values, so an unsorted frame silently produces
    garbage instead of an error. Centralized here so every feature
    function gets it for free instead of relying on the caller to
    remember.
    """
    return df.sort_values(["id", "date"]).reset_index(drop=True)


def add_lag_features(df: pd.DataFrame, lag_days: list[int] = LAG_DAYS) -> pd.DataFrame:
    """Adds `sales_lag_{n}` for each n in `lag_days`: the actual sales
    value from n days before, per series. `groupby("id")` keeps series
    from bleeding into each other at the boundary — without it, the first
    few rows of series B would incorrectly pick up trailing values from
    series A.
    """
    out = _sorted(df)
    grouped = out.groupby("id", sort=False)["sales"]
    for n in lag_days:
        out[f"sales_lag_{n}"] = grouped.shift(n)
    return out


def add_rolling_features(df: pd.DataFrame, windows: list[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    """Adds `sales_rollmean_{w}` and `sales_rollstd_{w}` for each window.

    The leakage-prevention step is `.shift(1)` BEFORE `.rolling(w)`: a
    naive `groupby("id")["sales"].rolling(w).mean()` would include the
    current day's own sales inside its own rolling average — the model
    would then partially be predicting sales from sales, which inflates
    backtest accuracy in a way that evaporates the moment it faces a real
    unseen day. Shifting first means the window for day t covers
    [t-w, t-1], never t itself.
    """
    out = _sorted(df)
    grouped = out.groupby("id", sort=False)["sales"]
    for w in windows:
        # transform (not a manual regroup-then-reset_index) is what keeps
        # this correctly aligned to `out`'s row order — an earlier draft
        # regrouped the already-shifted Series by id without sort=False,
        # which pandas can silently reorder alphabetically by group,
        # scrambling every row's feature value against the wrong id.
        # transform guarantees the result comes back in the original
        # frame's row order every time, which is why it's used here
        # instead of the more "obvious" shift-then-regroup pattern.
        out[f"sales_rollmean_{w}"] = grouped.transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
        )
        out[f"sales_rollstd_{w}"] = grouped.transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=2).std()
        )
    return out
