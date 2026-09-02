"""Price/promotion features derived from `sell_price`.

Not one of the two modules named in the day-1 stub (calendar.py, lags.py)
- added as its own file once it became clear price features are their own
separable concern (a different kind of derived signal from calendar/lag
features) rather than a natural fit inside either existing module.

Ingestion (`dsp.ingestion.load.enrich_with_prices`) already merges the raw
sell_prices table onto bronze on (store_id, item_id, wm_yr_wk) before
`BronzeSalesSchema` validates it - so by the time a frame reaches this
module, `sell_price` is already a real column, nullable where M5 genuinely
has no listed price for that store/item/week (not zero-filled - see
BronzeSalesSchema's docstring). This module used to also do that merge
itself (a `merge_prices` function, now removed) from a time before it was
clear ingestion already owned that step; keeping two independent price
merges around was a real, if latent, bug waiting to happen - the moment
`add_price_features` ran on a bronze frame that already had `sell_price`,
a second merge would collide with it, not add anything.
"""

from __future__ import annotations

import pandas as pd

# A rolling window over which "the regular price" is estimated as the
# max price seen - a real assumption, not a measured constant: 90 days
# (~13 weeks) is long enough to look past a short-lived promotional dip
# back to a plausible "normal" price, without reaching so far back that
# a genuine, permanent price change gets mistaken for an ongoing discount.
DEFAULT_PRICE_ROLLING_WINDOW_DAYS = 90


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

    Requires `sell_price` to already be a column on `df` (ingestion's job,
    not this function's - see module docstring). Raises rather than
    silently computing NaN-filled price features from a missing column,
    which would look like "no price data" instead of "this frame never
    went through ingestion's price merge."
    """
    if "sell_price" not in df.columns:
        raise ValueError(
            "add_price_features: 'sell_price' column not found - this function expects "
            "prices already merged into bronze by dsp.ingestion.load.enrich_with_prices, "
            "not merged here."
        )
    out = df.sort_values([id_col, date_col]).copy()

    price_groups = out.groupby(id_col, observed=True)["sell_price"]
    rolling_max = price_groups.rolling(window_days, min_periods=1).max()
    out["price_rolling_max"] = rolling_max.reset_index(level=0, drop=True)

    out["is_discounted"] = (out["sell_price"] < out["price_rolling_max"]).astype(int)
    out["price_change"] = out.groupby(id_col, observed=True)["sell_price"].diff()

    return out
