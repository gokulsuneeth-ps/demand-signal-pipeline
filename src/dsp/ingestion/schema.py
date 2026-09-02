"""Pandera schemas for the M5 raw tables and the bronze long-format output.

Two levels of validation on purpose:
  - Raw schemas (loose) catch a corrupted/incompatible download early, before
    any reshaping happens — better to fail on `load_calendar()` than three
    functions deep inside a merge.
  - The bronze schema (strict) is the contract the rest of the pipeline
    relies on. Day 3's feature functions are written assuming this schema
    holds, so if it doesn't, they should never see the data at all.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series


class CalendarSchema(pa.DataFrameModel):
    """`calendar.csv` as shipped by the M5 competition, loosely checked."""

    date: Series[str]
    wm_yr_wk: Series[int] = pa.Field(ge=11101)
    weekday: Series[str]
    wday: Series[int] = pa.Field(ge=1, le=7)
    month: Series[int] = pa.Field(ge=1, le=12)
    year: Series[int] = pa.Field(ge=2011, le=2016)
    d: Series[str] = pa.Field(str_matches=r"^d_\d+$")
    event_name_1: Series[str] = pa.Field(nullable=True)
    event_type_1: Series[str] = pa.Field(nullable=True)
    event_name_2: Series[str] = pa.Field(nullable=True)
    event_type_2: Series[str] = pa.Field(nullable=True)
    snap_CA: Series[int] = pa.Field(isin=[0, 1])
    snap_TX: Series[int] = pa.Field(isin=[0, 1])
    snap_WI: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        coerce = True


class SellPricesSchema(pa.DataFrameModel):
    """`sell_prices.csv` as shipped by the M5 competition."""

    store_id: Series[str]
    item_id: Series[str]
    wm_yr_wk: Series[int] = pa.Field(ge=11101)
    sell_price: Series[float] = pa.Field(ge=0)

    class Config:
        coerce = True


class BronzeSalesSchema(pa.DataFrameModel):
    """The long-format table ingestion produces — the contract every
    downstream stage (features, backtest, model) is written against.

    `sales` must be a non-negative integer: a demand series with negative
    units sold is a data bug, not a valid observation, and should fail loud
    here rather than quietly poison a lag feature three stages downstream.
    `sell_price` is nullable — an item can legitimately have no listed price
    for a given week (e.g. before it was carried at that store).
    """

    id: Series[str]
    item_id: Series[str]
    dept_id: Series[str]
    cat_id: Series[str]
    store_id: Series[str]
    state_id: Series[str]
    d: Series[str] = pa.Field(str_matches=r"^d_\d+$")
    sales: Series[int] = pa.Field(ge=0)
    date: Series[str]
    wm_yr_wk: Series[int] = pa.Field(ge=11101)
    wday: Series[int] = pa.Field(ge=1, le=7)
    month: Series[int] = pa.Field(ge=1, le=12)
    year: Series[int] = pa.Field(ge=2011, le=2016)
    event_name_1: Series[str] = pa.Field(nullable=True)
    event_type_1: Series[str] = pa.Field(nullable=True)
    event_name_2: Series[str] = pa.Field(nullable=True)
    event_type_2: Series[str] = pa.Field(nullable=True)
    snap_CA: Series[int] = pa.Field(isin=[0, 1])
    snap_TX: Series[int] = pa.Field(isin=[0, 1])
    snap_WI: Series[int] = pa.Field(isin=[0, 1])
    sell_price: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        coerce = True
        strict = True  # reject unexpected columns — schema drift should fail loudly
