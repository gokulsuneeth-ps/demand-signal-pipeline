"""Price/promotion features: merging in the raw sell_prices table, and
deriving discount/price-change signals from it.

Not one of the two modules named in the day-1 stub (calendar.py, lags.py)
- added as its own file, day 8, once it became clear price features are
their own separable concern (a different raw source table, a different
join key, a different kind of derived signal) rather than a natural fit
inside either existing module.
"""

from __future__ import annotations

import pandas as pd

# A rolling window over which "the regular price" is estimated as the
# max price seen - a real assumption, not a measured constant: 90 days
# (~13 weeks) is long enough to look past a short-lived promotional dip
# back to a plausible "normal" price, without reaching so far back that
# a genuine, permanent price change gets mistaken for an ongoing discount.
DEFAULT_PRICE_ROLLING_WINDOW_DAYS = 90


def merge_prices(
    bronze_df: pd.DataFrame, prices_df: pd.DataFrame, id_col: str = "id"
) -> pd.DataFrame:
    """Joins the raw sell_prices table onto bronze on (store_id, item_id,
    wm_yr_wk) - prices in M5 are set per WEEK, not per day, so multiple
    daily rows share one price row; this is a many-to-one merge, not
    one-to-one.

    A missing price after the merge (NaN `sell_price`) is left as NaN,
    not zero-filled or forward-filled - a real, common M5 data property
    (an item with no recorded price for a week genuinely wasn't being
    sold that week), and silently filling it would misrepresent "not
    sold" as "sold for free."

    Raises if the merge changes bronze_df's row count - a many-to-one
    merge should never multiply rows; if it does, sell_prices.csv has an
    unexpected duplicate (store_id, item_id, wm_yr_wk) combination, and
    that is a data-quality problem worth surfacing immediately, not
    silently absorbing as duplicated daily sales rows.
    """
    before = len(bronze_df)
    merged = bronze_df.merge(
        prices_df[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    if len(merged) != before:
        raise ValueError(
            f"merge_prices: row count changed from {before} to {len(merged)} - "
            f"sell_prices.csv likely has a duplicate (store_id, item_id, wm_yr_wk) "
            f"combination, which should be a one-price-per-week, per-item, per-store table"
        )
    return merged


def add_price_features(
    df: pd.DataFrame,
    window_days: int = DEFAULT_PRICE_ROLLING_WINDOW_DAYS,
    id_col: str = "id",
    date_col: str = "date",
) -> pd.DataFrame:
    """Adds `price_rolling_max` (a trailing proxy for "regular price" -
    see DEFAULT_PRICE_ROLLING_WINDOW_DAYS), `is_discounted` (today's
    price below that regular-price proxy), and `price_change` (day-over-
    day price difference for that series).

    `price_rolling_max` INCLUDES the current day's own price (unlike the
    sales rolling stats in lags.py) - this is a deliberate, different
    choice, not an inconsistency: a price is known and observable AT the
    moment it's charged (there's no leakage risk equivalent to "we don't
    yet know today's sales"), so excluding today's own price would only
    make `is_discounted` less accurate for no safety benefit. `sales`
    rolling stats exclude today because today's sales is the very thing
    being forecast; today's price is not.

    `price_change` and the rolling max both require rows sorted by
    (id_col, date_col) per series - enforced here defensively, same as
    lags.py.
    """
    out = df.sort_values([id_col, date_col]).copy()

    price_groups = out.groupby(id_col, observed=True)["sell_price"]
    rolling_max = price_groups.rolling(window_days, min_periods=1).max()
    out["price_rolling_max"] = rolling_max.reset_index(level=0, drop=True)

    out["is_discounted"] = (out["sell_price"] < out["price_rolling_max"]).astype(int)
    out["price_change"] = out.groupby(id_col, observed=True)["sell_price"].diff()

    return out
