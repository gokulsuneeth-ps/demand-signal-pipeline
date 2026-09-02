"""Tests for the calendar, lag/rolling, and price feature functions -
hand-calculated values, and dedicated leakage checks for the rolling
functions (the single most common bug class this kind of pipeline has,
per lags.py's own module docstring).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dsp.features.calendar import (
    add_calendar_features,
    add_is_event,
    add_is_snap_day,
    add_is_weekend,
)
from dsp.features.lags import add_lag_features, add_rolling_features
from dsp.features.prices import add_price_features, merge_prices

# --- calendar features ---


def test_add_is_weekend_uses_m5_saturday_sunday_convention():
    df = pd.DataFrame({"wday": [1, 2, 3, 4, 5, 6, 7]})
    out = add_is_weekend(df)
    assert out["is_weekend"].tolist() == [1, 1, 0, 0, 0, 0, 0]


def test_add_is_event_true_if_either_event_slot_set():
    df = pd.DataFrame(
        {
            "event_name_1": ["NewYear", None, None],
            "event_name_2": [None, "Purim", None],
        }
    )
    out = add_is_event(df)
    assert out["is_event"].tolist() == [1, 1, 0]


def test_add_is_snap_day_reads_the_row_own_state_column():
    df = pd.DataFrame(
        {
            "state_id": ["CA", "TX", "CA", "WI"],
            "snap_CA": [1, 1, 0, 1],
            "snap_TX": [0, 1, 0, 0],
            "snap_WI": [0, 0, 0, 0],
        }
    )
    out = add_is_snap_day(df)
    # row0: CA -> snap_CA=1 ; row1: TX -> snap_TX=1 ; row2: CA -> snap_CA=0 ; row3: WI -> snap_WI=0
    assert out["is_snap_day"].tolist() == [1, 1, 0, 0]


def test_add_is_snap_day_raises_on_state_with_no_matching_column():
    df = pd.DataFrame({"state_id": ["XX"], "snap_CA": [1]})
    with pytest.raises(ValueError, match="no snap_<STATE> column"):
        add_is_snap_day(df)


def test_add_calendar_features_applies_all_three():
    df = pd.DataFrame(
        {
            "wday": [1],
            "event_name_1": [None],
            "event_name_2": [None],
            "state_id": ["CA"],
            "snap_CA": [1],
        }
    )
    out = add_calendar_features(df)
    assert {"is_weekend", "is_event", "is_snap_day"}.issubset(out.columns)


# --- lag features ---


def test_add_lag_features_matches_hand_calculation():
    dates = pd.date_range("2016-01-01", periods=10)
    df = pd.DataFrame({"id": ["A"] * 10, "date": dates, "sales": list(range(1, 11))})
    out = add_lag_features(df, lags=(3,))
    # day index 7 (0-based) has value 8; sales_lag_3 should be day index 4's value = 5
    assert out.iloc[7]["sales_lag_3"] == 5
    assert pd.isna(out.iloc[0]["sales_lag_3"])  # not enough history yet


def test_add_lag_features_does_not_cross_series_boundaries():
    dates = pd.date_range("2016-01-01", periods=5)
    df = pd.DataFrame(
        {
            "id": ["A"] * 5 + ["B"] * 5,
            "date": list(dates) * 2,
            "sales": [10, 11, 12, 13, 14, 100, 101, 102, 103, 104],
        }
    )
    out = add_lag_features(df, lags=(2,))
    # series B's first two rows must NOT pull a lag value from series A's tail
    b_rows = out[out["id"] == "B"].sort_values("date")
    assert pd.isna(b_rows.iloc[0]["sales_lag_2"])
    assert pd.isna(b_rows.iloc[1]["sales_lag_2"])


# --- rolling features: correctness AND leakage ---


def test_add_rolling_features_matches_hand_calculation():
    dates = pd.date_range("2016-01-01", periods=10)
    df = pd.DataFrame({"id": ["A"] * 10, "date": dates, "sales": list(range(1, 11))})
    out = add_rolling_features(df, windows=(3,))
    # day index 5 (value 6): trailing 3 days EXCLUDING today = days 2,3,4 (values 3,4,5) -> mean 4
    assert out.iloc[5]["sales_rollmean_3"] == pytest.approx(4.0)
    assert out.iloc[5]["sales_rollstd_3"] == pytest.approx(np.std([3, 4, 5], ddof=1))


def test_add_rolling_features_excludes_current_day_no_leakage():
    """The critical regression test this module's docstring promises:
    changing ONLY today's own sales value must NOT change today's own
    rolling mean/std - if it did, the rolling stat would be leaking
    today's answer into today's own feature row.
    """
    dates = pd.date_range("2016-01-01", periods=10)
    baseline = pd.DataFrame({"id": ["A"] * 10, "date": dates, "sales": [5] * 10})
    mutated = baseline.copy()
    mutated.loc[mutated.index[-1], "sales"] = 999999  # blow up the LAST day's own value only

    out_baseline = add_rolling_features(baseline, windows=(3,))
    out_mutated = add_rolling_features(mutated, windows=(3,))

    last_row_baseline = out_baseline.iloc[-1]
    last_row_mutated = out_mutated.iloc[-1]
    assert last_row_baseline["sales_rollmean_3"] == pytest.approx(
        last_row_mutated["sales_rollmean_3"]
    )

    baseline_std = last_row_baseline["sales_rollstd_3"]
    mutated_std = last_row_mutated["sales_rollstd_3"]
    if pd.isna(baseline_std) or pd.isna(mutated_std):
        assert pd.isna(baseline_std) and pd.isna(mutated_std)
    else:
        assert baseline_std == pytest.approx(mutated_std)


def test_add_rolling_features_nan_until_full_window_available():
    dates = pd.date_range("2016-01-01", periods=5)
    df = pd.DataFrame({"id": ["A"] * 5, "date": dates, "sales": [1, 2, 3, 4, 5]})
    out = add_rolling_features(df, windows=(3,))
    # day index 2 (3rd day): only 2 prior days exist (indices 0,1) - not
    # a full window of 3 - must be NaN, not a "partial" average.
    assert pd.isna(out.iloc[2]["sales_rollmean_3"])
    # day index 3 (4th day): exactly 3 prior days exist (0,1,2) - full window.
    assert out.iloc[3]["sales_rollmean_3"] == pytest.approx(2.0)  # mean(1,2,3)


# --- price features ---


def test_merge_prices_joins_on_store_item_week():
    bronze = pd.DataFrame(
        {
            "id": ["A"] * 2,
            "store_id": ["CA_1"] * 2,
            "item_id": ["FOODS_1_001"] * 2,
            "wm_yr_wk": [11101, 11102],
        }
    )
    prices = pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1"],
            "item_id": ["FOODS_1_001", "FOODS_1_001"],
            "wm_yr_wk": [11101, 11102],
            "sell_price": [3.98, 4.48],
        }
    )
    out = merge_prices(bronze, prices)
    assert out["sell_price"].tolist() == [3.98, 4.48]


def test_merge_prices_leaves_missing_price_as_nan_not_zero():
    bronze = pd.DataFrame(
        {"id": ["A"], "store_id": ["CA_1"], "item_id": ["FOODS_1_001"], "wm_yr_wk": [99999]}
    )
    prices = pd.DataFrame(
        {
            "store_id": ["CA_1"],
            "item_id": ["FOODS_1_001"],
            "wm_yr_wk": [11101],
            "sell_price": [3.98],
        }
    )
    out = merge_prices(bronze, prices)
    assert pd.isna(out["sell_price"].iloc[0])


def test_merge_prices_raises_on_duplicate_price_rows():
    bronze = pd.DataFrame(
        {"id": ["A"], "store_id": ["CA_1"], "item_id": ["FOODS_1_001"], "wm_yr_wk": [11101]}
    )
    duplicated_prices = pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1"],
            "item_id": ["FOODS_1_001", "FOODS_1_001"],
            "wm_yr_wk": [11101, 11101],
            "sell_price": [3.98, 4.48],  # two different prices for the same week - bad data
        }
    )
    with pytest.raises(ValueError, match="row count changed"):
        merge_prices(bronze, duplicated_prices)


def test_add_price_features_matches_hand_calculation():
    dates = pd.date_range("2016-01-01", periods=4)
    df = pd.DataFrame({"id": ["A"] * 4, "date": dates, "sell_price": [5.0, 5.0, 3.0, 3.0]})
    out = add_price_features(df, window_days=4)
    # regular price proxy is the rolling max INCLUDING today -> 5.0 throughout
    assert out["price_rolling_max"].tolist() == [5.0, 5.0, 5.0, 5.0]
    # discounted on the days priced below that regular-price proxy
    assert out["is_discounted"].tolist() == [0, 0, 1, 1]
    # price_change: NaN, 0.0, -2.0, 0.0
    assert pd.isna(out["price_change"].iloc[0])
    assert out["price_change"].iloc[2] == pytest.approx(-2.0)
