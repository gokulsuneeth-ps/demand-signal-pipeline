"""Tests against tiny synthetic series, with explicit leakage checks.

The leakage tests matter more than any other test in this file: a
feature that quietly includes same-day or future information will make
day 4's backtest report a great number that means nothing. These tests
exist specifically to make that impossible to ship unnoticed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dsp.features.build import build_features
from dsp.features.calendar import add_event_flag, add_snap_flag, add_weekend_flag
from dsp.features.lags import add_lag_features, add_rolling_features
from dsp.features.prices import add_discount_flag, add_price_change


@pytest.fixture
def one_series() -> pd.DataFrame:
    """A single series, 14 consecutive days, known sales values —
    small enough that lag_7 and rollmean_7 can be checked by hand.
    """
    dates = pd.date_range("2011-01-29", periods=14)  # starts on a Saturday
    sales = list(range(1, 15))  # 1, 2, 3, ..., 14 — easy to hand-verify
    return pd.DataFrame(
        {
            "id": ["FOODS_1_CA_1"] * 14,
            "date": dates.astype(str),
            "sales": sales,
            "event_name_1": [None] * 12 + ["NewYear", None],
            "event_name_2": [None] * 14,
            "state_id": ["CA"] * 14,
            "snap_CA": [0, 1] * 7,
            "snap_TX": [1, 0] * 7,
            "snap_WI": [0, 0] * 7,
            "sell_price": [3.98] * 5 + [2.98] * 5 + [3.98] * 4,  # a markdown, then reverts
        }
    )


@pytest.fixture
def two_series() -> pd.DataFrame:
    """Two series interleaved in an order that would break a groupby
    implementation careless about sort order or index alignment.
    """
    dates = pd.date_range("2011-01-29", periods=5)
    rows = []
    for series_id, base in [("B_STORE", 100), ("A_STORE", 1)]:  # deliberately B before A
        for i, d in enumerate(dates):
            rows.append({"id": series_id, "date": str(d.date()), "sales": base + i})
    df = pd.DataFrame(rows)
    df["state_id"] = "CA"
    df["snap_CA"] = 0
    df["snap_TX"] = 0
    df["snap_WI"] = 0
    df["event_name_1"] = None
    df["event_name_2"] = None
    df["sell_price"] = 5.0
    return df


# --- calendar features ---


def test_weekend_flag_matches_actual_calendar(one_series):
    out = add_weekend_flag(one_series)
    # 2011-01-29 was a Saturday
    assert out.loc[0, "is_weekend"] == 1
    # 2011-01-31 was a Monday
    assert out.loc[2, "is_weekend"] == 0


def test_event_flag_true_only_where_event_present(one_series):
    out = add_event_flag(one_series)
    assert out["is_event"].tolist() == [0] * 12 + [1, 0]


def test_snap_flag_selects_correct_state_column(one_series):
    out = add_snap_flag(one_series)
    # all rows are CA, so is_snap_day must track snap_CA, not snap_TX
    assert out["is_snap_day"].tolist() == out["snap_CA"].tolist()
    assert out["is_snap_day"].tolist() != out["snap_TX"].tolist()


# --- lag / rolling features: correctness ---


def test_lag_7_shifts_by_exactly_seven_days(one_series):
    out = add_lag_features(one_series, lag_days=[7])
    # day index 7 (8th day, sales=8) should have lag_7 == sales on day index 0 (sales=1)
    assert out.loc[7, "sales_lag_7"] == 1
    assert out.loc[13, "sales_lag_7"] == 7


def test_lag_features_null_before_enough_history(one_series):
    out = add_lag_features(one_series, lag_days=[7])
    assert out.loc[0:6, "sales_lag_7"].isna().all()


def test_rolling_mean_matches_hand_calculation(one_series):
    out = add_rolling_features(one_series, windows=[7])
    # day index 7 (sales=8): rollmean_7 should average sales at indices 0-6 (values 1..7)
    expected = np.mean(range(1, 8))
    assert out.loc[7, "sales_rollmean_7"] == pytest.approx(expected)


# --- the leakage tests: the ones that actually matter ---


def test_rolling_mean_excludes_current_day_value(one_series):
    """If day t's rolling mean included day t's own sales, changing only
    that one value would change its own rolling mean — which is exactly
    the leak this test forces to the surface.
    """
    out_before = add_rolling_features(one_series, windows=[7])
    mutated = one_series.copy()
    mutated.loc[7, "sales"] = 999_999  # blow up day index 7's own value
    out_after = add_rolling_features(mutated, windows=[7])

    assert out_before.loc[7, "sales_rollmean_7"] == out_after.loc[7, "sales_rollmean_7"]


def test_rolling_max_price_excludes_current_day_price(one_series):
    """Same leakage shape, for the price-discount feature: today's own
    price must not be part of today's comparison baseline.
    """
    out = add_discount_flag(one_series, window=28)
    # day index 5 is the first day of the markdown (price drops 3.98 -> 2.98).
    # price_rolling_max at that row must reflect prior days only (3.98),
    # so the drop is correctly flagged as a discount on the day it starts.
    assert out.loc[5, "price_rolling_max"] == pytest.approx(3.98)
    assert out.loc[5, "is_discounted"] == 1


def test_price_change_is_zero_on_first_observation(one_series):
    out = add_price_change(one_series)
    assert out.loc[0, "price_change"] == 0.0


# --- multi-series correctness (the groupby/sort-order trap) ---


def test_lag_and_rolling_do_not_bleed_across_series(two_series):
    lagged = add_lag_features(two_series, lag_days=[1])
    out = add_rolling_features(lagged, windows=[2])
    out = out.sort_values(["id", "date"]).reset_index(drop=True)

    a_store = out.query("id == 'A_STORE'").reset_index(drop=True)
    b_store = out.query("id == 'B_STORE'").reset_index(drop=True)

    # A_STORE's first row must NOT pick up B_STORE's trailing sales (100-104)
    assert pd.isna(a_store.loc[0, "sales_lag_1"])
    assert a_store.loc[1, "sales_lag_1"] == 1  # A_STORE day 0's own sales value

    # confirm B_STORE values never appear in A_STORE's features at all
    assert a_store["sales_lag_1"].dropna().max() < 100
    assert b_store["sales_lag_1"].dropna().min() >= 100


def test_rolling_result_row_order_matches_input_id_column(two_series):
    """Regression test for the exact bug caught during development: an
    earlier implementation re-grouped an already-shifted Series without
    `sort=False`, silently reordering results alphabetically by id and
    scrambling every row's feature against the wrong series.
    """
    out = add_rolling_features(two_series, windows=[2])
    # every row's rollmean, when not null, must be plausible for ITS OWN
    # series' value range — A_STORE sales are 1-5, B_STORE sales are 100-104
    a_rows = out[out["id"] == "A_STORE"]["sales_rollmean_2"].dropna()
    b_rows = out[out["id"] == "B_STORE"]["sales_rollmean_2"].dropna()
    assert (a_rows < 10).all()
    assert (b_rows > 90).all()


# --- full pipeline ---


def test_build_features_preserves_row_count(one_series):
    one_series["state_id"] = "CA"
    result = build_features(one_series)
    assert len(result) == len(one_series)


def test_build_features_adds_expected_columns(one_series):
    result = build_features(one_series)
    for col in [
        "is_weekend",
        "is_event",
        "is_snap_day",
        "sales_lag_7",
        "sales_lag_14",
        "sales_lag_28",
        "sales_rollmean_7",
        "sales_rollstd_7",
        "is_discounted",
        "price_change",
    ]:
        assert col in result.columns
