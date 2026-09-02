"""Tests for build_bronze / read_calendar / read_prices, using small,
hand-built CSVs matching M5's real raw schema exactly (column names and
shapes copied from the actual downloaded files, not guessed) - including
deliberately malformed fixtures that must raise, not silently reshape
wrong.
"""

from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest
from dsp.ingestion.ingest import build_bronze, read_calendar, read_prices


def _write_calendar_csv(path, n_days: int = 10) -> None:
    dates = pd.date_range("2016-01-01", periods=n_days)
    rows = []
    for i, d in enumerate(dates):
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "wm_yr_wk": 11101 + i // 7,
                "weekday": d.day_name(),
                "wday": (d.dayofweek + 2) % 7 or 7,  # arbitrary but valid 1-7
                "month": d.month,
                "year": d.year,
                "d": f"d_{i + 1}",
                "event_name_1": "SuperBowl" if i == 3 else None,
                "event_type_1": "Sporting" if i == 3 else None,
                "event_name_2": None,
                "event_type_2": None,
                "snap_CA": int(i % 3 == 0),
                "snap_TX": 0,
                "snap_WI": 0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_sales_csv(path, n_days: int = 10) -> None:
    rows = [
        {
            "id": "FOODS_1_001_CA_1_evaluation",
            "item_id": "FOODS_1_001",
            "dept_id": "FOODS_1",
            "cat_id": "FOODS",
            "store_id": "CA_1",
            "state_id": "CA",
        },
        {
            "id": "HOBBIES_1_001_CA_1_evaluation",
            "item_id": "HOBBIES_1_001",
            "dept_id": "HOBBIES_1",
            "cat_id": "HOBBIES",
            "store_id": "CA_1",
            "state_id": "CA",
        },
        {
            "id": "FOODS_1_001_TX_1_evaluation",
            "item_id": "FOODS_1_001",
            "dept_id": "FOODS_1",
            "cat_id": "FOODS",
            "store_id": "TX_1",
            "state_id": "TX",
        },
    ]
    df = pd.DataFrame(rows)
    for i in range(n_days):
        df[f"d_{i + 1}"] = [3, 1, 5][: len(df)]
    df.to_csv(path, index=False)


@pytest.fixture
def raw_csvs(tmp_path):
    calendar_path = tmp_path / "calendar.csv"
    sales_path = tmp_path / "sales_train_evaluation.csv"
    _write_calendar_csv(calendar_path)
    _write_sales_csv(sales_path)
    return calendar_path, sales_path


def test_build_bronze_filters_to_state_and_category(raw_csvs):
    calendar_path, sales_path = raw_csvs
    bronze = build_bronze(
        str(calendar_path), str(sales_path), state_filter=["CA"], category_filter=["FOODS"]
    )
    # Only FOODS_1_001_CA_1 matches BOTH filters - HOBBIES is wrong
    # category, TX is wrong state.
    assert set(bronze["id"].unique()) == {"FOODS_1_001_CA_1_evaluation"}


def test_build_bronze_keeps_everything_when_filters_are_none(raw_csvs):
    calendar_path, sales_path = raw_csvs
    bronze = build_bronze(
        str(calendar_path), str(sales_path), state_filter=None, category_filter=None
    )
    assert set(bronze["id"].unique()) == {
        "FOODS_1_001_CA_1_evaluation",
        "HOBBIES_1_001_CA_1_evaluation",
        "FOODS_1_001_TX_1_evaluation",
    }


def test_build_bronze_produces_one_row_per_series_per_day(raw_csvs):
    calendar_path, sales_path = raw_csvs
    bronze = build_bronze(
        str(calendar_path), str(sales_path), state_filter=None, category_filter=None
    )
    assert len(bronze) == 3 * 10  # 3 series x 10 days
    assert not bronze.duplicated(subset=["id", "date"]).any()


def test_build_bronze_preserves_real_sales_values(raw_csvs):
    calendar_path, sales_path = raw_csvs
    bronze = build_bronze(
        str(calendar_path), str(sales_path), state_filter=None, category_filter=None
    )
    foods_row = bronze[bronze["id"] == "FOODS_1_001_CA_1_evaluation"].iloc[0]
    assert foods_row["sales"] == 3


def test_build_bronze_raises_on_negative_sales(tmp_path):
    """The schema's ge(0) check on 'sales' must actually fire - a
    negative sales value is a real data-quality problem (a bad
    correction, a reshape bug) that must never silently become a
    negative "demand" number feeding every downstream metric.
    """
    calendar_path = tmp_path / "calendar.csv"
    sales_path = tmp_path / "sales_train_evaluation.csv"
    _write_calendar_csv(calendar_path)

    df = pd.DataFrame(
        [
            {
                "id": "FOODS_1_001_CA_1_evaluation",
                "item_id": "FOODS_1_001",
                "dept_id": "FOODS_1",
                "cat_id": "FOODS",
                "store_id": "CA_1",
                "state_id": "CA",
                "d_1": -5,
            }
        ]
    )
    for i in range(1, 10):
        df[f"d_{i + 1}"] = 1
    df.to_csv(sales_path, index=False)

    with pytest.raises(pandera.errors.SchemaError):
        build_bronze(str(calendar_path), str(sales_path), state_filter=None, category_filter=None)


def test_build_bronze_raises_on_malformed_wide_columns(tmp_path):
    calendar_path = tmp_path / "calendar.csv"
    sales_path = tmp_path / "sales_train_evaluation.csv"
    _write_calendar_csv(calendar_path)

    bad_df = pd.DataFrame(
        [
            {
                "id": "FOODS_1_001_CA_1_evaluation",
                "item_id": "FOODS_1_001",
                "dept_id": "FOODS_1",
                "cat_id": "FOODS",
                "store_id": "CA_1",
                "state_id": "CA",
                "not_a_day_column": 3,
            }
        ]
    )
    bad_df.to_csv(sales_path, index=False)

    with pytest.raises(ValueError, match="expected wide shape"):
        build_bronze(str(calendar_path), str(sales_path), state_filter=None, category_filter=None)


def test_read_calendar_validates_schema(tmp_path):
    calendar_path = tmp_path / "calendar.csv"
    _write_calendar_csv(calendar_path)
    calendar = read_calendar(str(calendar_path))
    assert len(calendar) == 10
    assert pd.api.types.is_datetime64_any_dtype(calendar["date"])


def test_read_prices_returns_raw_shape(tmp_path):
    prices_path = tmp_path / "sell_prices.csv"
    pd.DataFrame(
        [
            {"store_id": "CA_1", "item_id": "FOODS_1_001", "wm_yr_wk": 11101, "sell_price": 3.98},
        ]
    ).to_csv(prices_path, index=False)
    prices = read_prices(str(prices_path))
    assert list(prices.columns) == ["store_id", "item_id", "wm_yr_wk", "sell_price"]
