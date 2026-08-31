"""Tests against tiny synthetic fixtures, not the real M5 download.

This is deliberate: CI has no Kaggle credentials and shouldn't need any —
these fixtures are small enough to read in one glance and exercise exactly
the logic that matters (filtering, reshaping, join correctness, and that
bad data actually fails validation instead of silently passing through).
"""

from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest
from dsp.ingestion.load import (
    ID_COLS,
    enrich_with_calendar,
    enrich_with_prices,
    filter_subset,
    melt_to_long,
)
from dsp.ingestion.schema import BronzeSalesSchema


@pytest.fixture
def sales_wide() -> pd.DataFrame:
    """4 series: 2 FOODS/CA (should survive filtering), 1 FOODS/TX (wrong
    state), 1 HOUSEHOLD/CA (wrong category) — so filter_subset has both a
    state and a category dimension to actually filter on, not just one.
    """
    return pd.DataFrame(
        {
            "id": ["FOODS_1_CA_1", "FOODS_2_CA_1", "FOODS_1_TX_1", "HOUSEHOLD_1_CA_1"],
            "item_id": ["FOODS_1", "FOODS_2", "FOODS_1", "HOUSEHOLD_1"],
            "dept_id": ["FOODS_1", "FOODS_2", "FOODS_1", "HOUSEHOLD_1"],
            "cat_id": ["FOODS", "FOODS", "FOODS", "HOUSEHOLD"],
            "store_id": ["CA_1", "CA_1", "TX_1", "CA_1"],
            "state_id": ["CA", "CA", "TX", "CA"],
            "d_1": [3, 0, 5, 1],
            "d_2": [1, 2, 4, 0],
            "d_3": [0, 1, 2, 2],
        }
    )


@pytest.fixture
def calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2011-01-29", "2011-01-30", "2011-01-31"],
            "wm_yr_wk": [11101, 11101, 11101],
            "weekday": ["Saturday", "Sunday", "Monday"],
            "wday": [1, 2, 3],
            "month": [1, 1, 1],
            "year": [2011, 2011, 2011],
            "d": ["d_1", "d_2", "d_3"],
            "event_name_1": [None, None, "NewYear"],
            "event_type_1": [None, None, "National"],
            "event_name_2": [None, None, None],
            "event_type_2": [None, None, None],
            "snap_CA": [0, 1, 1],
            "snap_TX": [1, 1, 0],
            "snap_WI": [0, 0, 0],
        }
    )


@pytest.fixture
def prices() -> pd.DataFrame:
    # Deliberately missing a price for FOODS_2 at CA_1 — exercises the
    # left-join-produces-a-null-price path that BronzeSalesSchema allows.
    return pd.DataFrame(
        {
            "store_id": ["CA_1"],
            "item_id": ["FOODS_1"],
            "wm_yr_wk": [11101],
            "sell_price": [3.98],
        }
    )


def test_filter_subset_keeps_only_foods_ca(sales_wide):
    result = filter_subset(sales_wide, cat_id="FOODS", state_id="CA")
    assert set(result["id"]) == {"FOODS_1_CA_1", "FOODS_2_CA_1"}


def test_filter_subset_raises_on_empty_result(sales_wide):
    with pytest.raises(ValueError, match="No rows matched"):
        filter_subset(sales_wide, cat_id="TOYS", state_id="CA")


def test_melt_to_long_row_count_and_shape(sales_wide):
    subset = filter_subset(sales_wide, cat_id="FOODS", state_id="CA")
    long_df = melt_to_long(subset)
    # 2 series x 3 days = 6 rows
    assert len(long_df) == 6
    assert set(long_df.columns) == set(ID_COLS) | {"d", "sales"}


def test_melt_to_long_preserves_values(sales_wide):
    subset = filter_subset(sales_wide, cat_id="FOODS", state_id="CA")
    long_df = melt_to_long(subset)
    row = long_df.query("id == 'FOODS_1_CA_1' and d == 'd_1'").iloc[0]
    assert row["sales"] == 3


def test_enrich_with_calendar_joins_on_d(sales_wide, calendar):
    subset = filter_subset(sales_wide, cat_id="FOODS", state_id="CA")
    long_df = melt_to_long(subset)
    enriched = enrich_with_calendar(long_df, calendar)
    row = enriched.query("id == 'FOODS_1_CA_1' and d == 'd_3'").iloc[0]
    assert row["date"] == "2011-01-31"
    assert row["event_name_1"] == "NewYear"
    assert row["snap_CA"] == 1


def test_enrich_with_prices_allows_missing_price(sales_wide, calendar, prices):
    subset = filter_subset(sales_wide, cat_id="FOODS", state_id="CA")
    long_df = melt_to_long(subset)
    enriched = enrich_with_calendar(long_df, calendar)
    priced = enrich_with_prices(enriched, prices)

    foods1_price = priced.query("id == 'FOODS_1_CA_1' and d == 'd_1'").iloc[0]["sell_price"]
    foods2_price = priced.query("id == 'FOODS_2_CA_1' and d == 'd_1'").iloc[0]["sell_price"]

    assert foods1_price == 3.98
    assert pd.isna(foods2_price)  # no matching price row — must stay null, not error


def test_bronze_schema_accepts_valid_pipeline_output(sales_wide, calendar, prices):
    subset = filter_subset(sales_wide, cat_id="FOODS", state_id="CA")
    long_df = melt_to_long(subset)
    enriched = enrich_with_calendar(long_df, calendar)
    bronze = enrich_with_prices(enriched, prices)

    validated = BronzeSalesSchema.validate(bronze)
    assert len(validated) == 6


def test_bronze_schema_rejects_negative_sales(sales_wide, calendar, prices):
    subset = filter_subset(sales_wide, cat_id="FOODS", state_id="CA")
    long_df = melt_to_long(subset)
    long_df.loc[0, "sales"] = -1  # inject a bad value
    enriched = enrich_with_calendar(long_df, calendar)
    bronze = enrich_with_prices(enriched, prices)

    with pytest.raises(pandera.errors.SchemaError):
        BronzeSalesSchema.validate(bronze)


def test_bronze_schema_rejects_unexpected_column(sales_wide, calendar, prices):
    subset = filter_subset(sales_wide, cat_id="FOODS", state_id="CA")
    long_df = melt_to_long(subset)
    enriched = enrich_with_calendar(long_df, calendar)
    bronze = enrich_with_prices(enriched, prices)
    bronze["unexpected_column"] = "surprise"

    # strict=True raises the aggregate SchemaErrors (plural), not the
    # single-failure SchemaError a Field check like ge=0 raises above.
    with pytest.raises(pandera.errors.SchemaErrors):
        BronzeSalesSchema.validate(bronze, lazy=True)
