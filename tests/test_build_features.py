"""Integration test for build_silver_features: end-to-end from bronze +
prices through to the final silver table, checking the exact set of
columns downstream code (dsp.models.train) actually depends on is
present, and that the pipeline steps compose correctly (a price feature
computed from a correctly-merged price, a lag computed from real sales).
"""

from __future__ import annotations

import pandas as pd
import pytest
from dsp.features.build import build_silver_features, build_silver_features_chunked

EXPECTED_SILVER_COLUMNS = {
    "id", "item_id", "dept_id", "store_id", "state_id", "cat_id",
    "date", "d", "wm_yr_wk", "sales",
    "wday", "month", "year", "sell_price",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    "snap_CA", "snap_TX", "snap_WI",
    "is_weekend", "is_event", "is_snap_day",
    "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_rollmean_7", "sales_rollstd_7", "sales_rollmean_28", "sales_rollstd_28",
    "price_rolling_max", "is_discounted", "price_change",
}  # fmt: skip


def _tiny_bronze_and_prices():
    dates = pd.date_range("2016-01-01", periods=35)
    bronze_rows = []
    for i, d in enumerate(dates):
        bronze_rows.append(
            {
                "id": "FOODS_1_001_CA_1_evaluation",
                "item_id": "FOODS_1_001",
                "dept_id": "FOODS_1",
                "cat_id": "FOODS",
                "store_id": "CA_1",
                "state_id": "CA",
                "date": d,
                "d": f"d_{i + 1}",
                "wm_yr_wk": 11101 + i // 7,
                "wday": (i % 7) + 1,
                "month": d.month,
                "year": d.year,
                "sales": (i % 5) + 1,
                "event_name_1": "SuperBowl" if i == 5 else None,
                "event_type_1": "Sporting" if i == 5 else None,
                "event_name_2": None,
                "event_type_2": None,
                "snap_CA": int(i % 3 == 0),
                "snap_TX": 0,
                "snap_WI": 0,
            }
        )
    bronze = pd.DataFrame(bronze_rows)

    weeks = sorted(bronze["wm_yr_wk"].unique())
    prices = pd.DataFrame(
        {
            "store_id": ["CA_1"] * len(weeks),
            "item_id": ["FOODS_1_001"] * len(weeks),
            "wm_yr_wk": weeks,
            "sell_price": [3.98 if w % 2 == 0 else 4.48 for w in weeks],
        }
    )
    return bronze, prices


def test_build_silver_features_produces_expected_columns():
    bronze, prices = _tiny_bronze_and_prices()
    silver = build_silver_features(bronze, prices)
    assert EXPECTED_SILVER_COLUMNS.issubset(set(silver.columns))


def test_build_silver_features_preserves_row_count():
    bronze, prices = _tiny_bronze_and_prices()
    silver = build_silver_features(bronze, prices)
    assert len(silver) == len(bronze)


def _two_store_bronze_and_prices():
    """Two series in two different stores - the minimal case that
    actually exercises chunking (a single-store fixture would pass
    `build_silver_features_chunked` trivially with exactly one chunk).
    """
    bronze_a, prices_a = _tiny_bronze_and_prices()

    bronze_b = bronze_a.copy()
    bronze_b["id"] = "FOODS_1_002_CA_2_evaluation"
    bronze_b["item_id"] = "FOODS_1_002"
    bronze_b["store_id"] = "CA_2"
    bronze_b["sales"] = bronze_b["sales"] + 10  # distinct values, not a copy-paste duplicate

    prices_b = prices_a.copy()
    prices_b["store_id"] = "CA_2"
    prices_b["item_id"] = "FOODS_1_002"
    prices_b["sell_price"] = prices_b["sell_price"] + 1.0

    bronze = pd.concat([bronze_a, bronze_b], ignore_index=True)
    prices = pd.concat([prices_a, prices_b], ignore_index=True)
    return bronze, prices


def test_build_silver_features_chunked_matches_unchunked():
    bronze, prices = _two_store_bronze_and_prices()

    unchunked = build_silver_features(bronze.copy(), prices.copy())
    chunked = build_silver_features_chunked(bronze.copy(), prices.copy(), chunk_col="store_id")

    unchunked = unchunked.sort_values(["id", "date"]).reset_index(drop=True)
    chunked = chunked.sort_values(["id", "date"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(unchunked, chunked, check_like=True)


def test_build_silver_features_chunked_handles_stale_category_list():
    """Regression test for a real bug hit against the actual M5 data:
    `id` is `category` dtype (see schema.py), and filtering a chunk's
    ROWS does not shrink its category LIST - a chunk built from a wider
    frame still carries every category that existed before filtering,
    most with zero rows in this chunk. Grouping by that column without
    `observed=True` iterates the empty categories too, and pandas'
    rolling-window code crashed outright on an empty group
    (`IndexError: index -1 is out of bounds for axis 0 with size 0` in
    `add_price_features`'s `.rolling(...)` call) - this only reproduces
    when the `id` column's category list is wider than what's actually
    present in the chunk being processed, which is exactly what this
    test sets up by building bronze from a 3-store universe and then
    only chunk-processing 2 of those stores' data.
    """
    bronze, prices = _two_store_bronze_and_prices()

    bronze_c, prices_c = _tiny_bronze_and_prices()
    bronze_c["id"] = "FOODS_1_003_CA_3_evaluation"
    bronze_c["item_id"] = "FOODS_1_003"
    bronze_c["store_id"] = "CA_3"
    prices_c["store_id"] = "CA_3"
    prices_c["item_id"] = "FOODS_1_003"

    full_bronze = pd.concat([bronze, bronze_c], ignore_index=True)
    for col in ["id", "item_id", "store_id"]:
        full_bronze[col] = full_bronze[col].astype("category")

    # Only pass the CA_3 rows through, but with `id`'s category dtype
    # still carrying all 3 stores' worth of categories - reproducing the
    # real stale-category-list shape without needing the full dataset.
    ca3_only_bronze = full_bronze[full_bronze["store_id"] == "CA_3"].copy()
    assert len(ca3_only_bronze["id"].cat.categories) > ca3_only_bronze["id"].nunique()

    result = build_silver_features_chunked(ca3_only_bronze, prices_c, chunk_col="store_id")
    assert result["id"].nunique() == 1
    assert len(result) == len(ca3_only_bronze)


def test_build_silver_features_chunked_raises_if_chunk_col_splits_an_id():
    bronze, prices = _tiny_bronze_and_prices()
    # `d` varies within every single id (that's the whole point of `d`) -
    # chunking on it would silently split every series' own history
    # across chunks, exactly the bug the safety check exists to catch.
    with pytest.raises(ValueError, match="span more than one"):
        build_silver_features_chunked(bronze, prices, chunk_col="d")


def test_build_silver_features_price_and_lag_values_are_real_not_placeholder():
    bronze, prices = _tiny_bronze_and_prices()
    silver = build_silver_features(bronze, prices).sort_values("date").reset_index(drop=True)

    assert silver["sell_price"].notna().all()
    assert set(silver["sell_price"].unique()) == {3.98, 4.48}

    # day index 28 (29th day): sales_lag_28 should equal day index 0's sales
    assert silver.iloc[28]["sales_lag_28"] == silver.iloc[0]["sales"]
