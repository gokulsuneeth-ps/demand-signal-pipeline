"""Price-derived features.

Unlike `sales` (the forecast target — see lags.py), `sell_price` is set
by the retailer in advance of the day it applies to. Today's price is
legitimately known at forecast time, so it's used same-day here, not
lagged — using yesterday's price would throw away real signal for no
leakage-prevention benefit.
"""

from __future__ import annotations

import pandas as pd

PRICE_ROLLING_WINDOW = 28


def add_discount_flag(df: pd.DataFrame, window: int = PRICE_ROLLING_WINDOW) -> pd.DataFrame:
    """`is_discounted`: today's price is below the trailing rolling-max
    price for this series. A rolling max (not the all-time max) so a
    price that's been stable-but-lower for months isn't flagged as a
    permanent "discount" — it only fires on an actual recent markdown.

    The rolling max itself IS shifted (via `.shift(1)` before `.rolling`)
    for the same reason sales features are: comparing today's price
    against a max that already includes today's price would bias toward
    never detecting a discount on the day it starts.
    """
    out = df.sort_values(["id", "date"]).reset_index(drop=True)
    grouped = out.groupby("id", sort=False)["sell_price"]
    rolling_max = grouped.transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).max())
    out["price_rolling_max"] = rolling_max
    out["is_discounted"] = (
        (out["sell_price"] < out["price_rolling_max"]) & out["sell_price"].notna()
    ).astype(int)
    return out


def add_price_change(df: pd.DataFrame) -> pd.DataFrame:
    """`price_change`: today's price minus yesterday's, per series. Zero
    for the first observation of a series (no prior price to diff
    against) and wherever price is null on either side.
    """
    out = df.sort_values(["id", "date"]).reset_index(drop=True)
    grouped = out.groupby("id", sort=False)["sell_price"]
    out["price_change"] = grouped.diff().fillna(0.0)
    return out


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_price_change(add_discount_flag(df))
