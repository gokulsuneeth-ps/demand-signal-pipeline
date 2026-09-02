"""Integration test for build_silver_features: end-to-end from bronze
(already carrying `sell_price` - ingestion's job, see build.py's module
docstring) through to the final silver table, checking the exact set of
columns downstream code (dsp.models.train) actually depends on is
present, and that the pipeline steps compose correctly (a price feature
computed from a real price, a lag computed from real sales).
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


def _tiny_bronze():
    """Bronze with `sell_price` already populated, matching what
    `dsp.ingestion.load.run_ingestion` actually hands to features - this
    module never merges prices itself (see build.py's module docstring).
    """
    dates = pd.date_range("2016-01-01", periods=35)
    bronze_rows = []
    for i, d in enumerate(dates):
        week = 11101 + i // 7
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
                "wm_yr_wk": week,
                "wday": (i % 7) + 1,
                "month": d.month,
                "year": d.year,
                "sales": (i % 5) + 1,
                "sell_price": 3.98 if week % 2 == 0 else 4.48,
                "event_name_1": "SuperBowl" if i == 5 else None,
                "event_type_1": "Sporting" if i == 5 else None,
                "event_name_2": None,
                "event_type_2": None,
                "snap_CA": int(i % 3 == 0),
                "snap_TX": 0,
                "snap_WI": 0,
            }
        )
    return pd.DataFrame(bronze_rows)


def test_build_silver_features_produces_expected_columns():
    bronze = _tiny_bronze()
    silver = build_silver_features(bronze)
    assert EXPECTED_SILVER_COLUMNS.issubset(set(silver.columns))


def test_build_silver_features_preserves_row_count():
    bronze = _tiny_bronze()
    silver = build_silver_features(bronze)
    assert len(silver) == len(bronze)


def test_build_silver_features_raises_if_sell_price_missing():
    """This module doesn't merge prices itself anymore - a bronze frame
    that skipped ingestion's price merge should fail loudly, not produce
    a silver table with NaN price features that look like "no discount
    data" instead of "this frame is missing a required column."
    """
    bronze = _tiny_bronze().drop(columns=["sell_price"])
    with pytest.raises(ValueError, match="sell_price"):
        build_silver_features(bronze)


def _two_store_bronze():
    """Two series in two different stores - the minimal case that
    actually exercises chunking (a single-store fixture would pass
    `build_silver_features_chunked` trivially with exactly one chunk).
    """
    bronze_a = _tiny_bronze()

    bronze_b = bronze_a.copy()
    bronze_b["id"] = "FOODS_1_002_CA_2_evaluation"
    bronze_b["item_id"] = "FOODS_1_002"
    bronze_b["store_id"] = "CA_2"
    bronze_b["sales"] = bronze_b["sales"] + 10  # distinct values, not a copy-paste duplicate
    bronze_b["sell_price"] = bronze_b["sell_price"] + 1.0

    return pd.concat([bronze_a, bronze_b], ignore_index=True)


def test_build_silver_features_chunked_matches_unchunked():
    bronze = _two_store_bronze()

    unchunked = build_silver_features(bronze.copy())
    chunked = build_silver_features_chunked(bronze.copy(), chunk_col="store_id")

    unchunked = unchunked.sort_values(["id", "date"]).reset_index(drop=True)
    chunked = chunked.sort_values(["id", "date"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(unchunked, chunked, check_like=True)


def test_build_silver_features_chunked_handles_stale_category_list():
    """Regression test for a real bug hit against the actual M5 data:
    when `id` is `category` dtype, filtering a chunk's ROWS does not
    shrink its category LIST - a chunk built from a wider frame still
    carries every category that existed before filtering, most with zero
    rows in this chunk. Grouping by that column without `observed=True`
    iterates the empty categories too, and pandas' rolling-window code
    crashed outright on an empty group (`IndexError: index -1 is out of
    bounds for axis 0 with size 0` in `add_price_features`'s
    `.rolling(...)` call) - this only reproduces when the `id` column's
    category list is wider than what's actually present in the chunk
    being processed, which is exactly what this test sets up by building
    bronze from a 3-store universe and then only chunk-processing 2 of
    those stores' data.

    Note: `dsp.ingestion.load`'s bronze is currently plain `str`/object
    dtype for `id` (see schema.py's `Series[str]`), not `category` - this
    test exists so the chunked path stays correct if/when a category-
    dtype memory optimization is added to ingestion later, not because
    it's required for correctness against today's bronze shape.
    """
    bronze = _two_store_bronze()

    bronze_c = _tiny_bronze()
    bronze_c["id"] = "FOODS_1_003_CA_3_evaluation"
    bronze_c["item_id"] = "FOODS_1_003"
    bronze_c["store_id"] = "CA_3"

    full_bronze = pd.concat([bronze, bronze_c], ignore_index=True)
    for col in ["id", "item_id", "store_id"]:
        full_bronze[col] = full_bronze[col].astype("category")

    # Only pass the CA_3 rows through, but with `id`'s category dtype
    # still carrying all 3 stores' worth of categories - reproducing the
    # real stale-category-list shape without needing the full dataset.
    ca3_only_bronze = full_bronze[full_bronze["store_id"] == "CA_3"].copy()
    assert len(ca3_only_bronze["id"].cat.categories) > ca3_only_bronze["id"].nunique()

    result = build_silver_features_chunked(ca3_only_bronze, chunk_col="store_id")
    assert result["id"].nunique() == 1
    assert len(result) == len(ca3_only_bronze)


def test_build_silver_features_chunked_raises_if_chunk_col_splits_an_id():
    bronze = _tiny_bronze()
    # `d` varies within every single id (that's the whole point of `d`) -
    # chunking on it would silently split every series' own history
    # across chunks, exactly the bug the safety check exists to catch.
    with pytest.raises(ValueError, match="span more than one"):
        build_silver_features_chunked(bronze, chunk_col="d")


def test_build_silver_features_price_and_lag_values_are_real_not_placeholder():
    bronze = _tiny_bronze()
    silver = build_silver_features(bronze).sort_values("date").reset_index(drop=True)

    assert silver["sell_price"].notna().all()
    assert set(silver["sell_price"].unique()) == {3.98, 4.48}

    # day index 28 (29th day): sales_lag_28 should equal day index 0's sales
    assert silver.iloc[28]["sales_lag_28"] == silver.iloc[0]["sales"]
