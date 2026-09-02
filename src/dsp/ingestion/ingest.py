"""Reads raw M5 CSVs (calendar, wide-format sales, sell prices) and
reshapes them into the validated, long-format bronze table every other
stage of this pipeline builds on.

Intentionally thin, matching docs/architecture.md's stated non-goal: this
is not a general-purpose ingestion framework, just the specific reshape
this project's raw data needs, with real validation at the boundary.
"""

from __future__ import annotations

import logging

import pandas as pd

from dsp.ingestion.schema import BronzeSalesSchema, CalendarSchema

logger = logging.getLogger(__name__)

# This project's deliberately narrowed scope (see PROBLEM_STATEMENT.md and
# the project explainer document): California only, FOODS category only.
# Named constants, not string literals scattered through the module, so
# the scope decision is visible and changeable in one place.
DEFAULT_STATE_FILTER = ["CA"]
DEFAULT_CATEGORY_FILTER = ["FOODS"]


def read_calendar(path: str) -> pd.DataFrame:
    """Reads calendar.csv and validates it against CalendarSchema.

    Coerces `date` to datetime before validation (the raw CSV stores it
    as plain text) - this is the one deliberate type conversion done
    before validation, everything else is checked as-is against what the
    CSV actually contains, so a genuinely malformed raw file still fails
    loudly here rather than after being silently reshaped.
    """
    calendar = pd.read_csv(path)
    calendar["date"] = pd.to_datetime(calendar["date"])
    return CalendarSchema.validate(calendar)


def read_prices(path: str) -> pd.DataFrame:
    """Reads sell_prices.csv as-is - no reshape needed, it's already one
    row per (store_id, item_id, wm_yr_wk).
    """
    return pd.read_csv(path)


def _melt_wide_sales(sales_wide: pd.DataFrame) -> pd.DataFrame:
    """Reshapes the raw wide-format sales table (one row per series, one
    column per day: d_1, d_2, ... d_1941) into long format (one row per
    series per day) - the shape every other stage of this pipeline
    expects.

    Raises if the wide file doesn't actually have the day-column shape
    this reshape assumes (e.g. a corrupted or truncated download) rather
    than silently melting whatever columns happen to exist and producing
    a bronze table with the wrong number of days per series.

    A real memory trap caught here during development: `pandas.melt`
    with the id columns left as plain strings repeats every one of those
    strings once per day column in the output - for ~5,700 series x
    ~1,900 days that inflated a sub-1GB wide table into a 5+GB long
    table and OOM-killed the process outright. Casting the id columns to
    `category` dtype BEFORE melting keeps each distinct string stored
    once, cutting the melted table's memory by roughly 6x in practice -
    the fix is applied here, at the one place the explosion actually
    happens, not by asking every caller to remember to do it themselves.
    """
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales_wide.columns if c not in id_cols]

    if not day_cols or not all(c.startswith("d_") for c in day_cols):
        raise ValueError(
            "sales_train_evaluation.csv does not have the expected wide "
            "shape (id/item_id/dept_id/cat_id/store_id/state_id followed "
            "only by d_1, d_2, ... day columns) - got unexpected columns: "
            f"{[c for c in day_cols if not c.startswith('d_')][:5]}"
        )

    sales_wide = sales_wide.copy()
    for col in id_cols:
        sales_wide[col] = sales_wide[col].astype("category")

    long = sales_wide.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales")
    return long


def build_bronze(
    calendar_path: str,
    sales_path: str,
    state_filter: list[str] | None = DEFAULT_STATE_FILTER,
    category_filter: list[str] | None = DEFAULT_CATEGORY_FILTER,
) -> pd.DataFrame:
    """Full raw -> bronze pipeline: read the wide sales file, melt it to
    long format, join in calendar.csv's per-day columns (date, wday,
    month, year, event/SNAP flags), apply the project's scope filter, and
    validate the result against BronzeSalesSchema before returning it.

    `state_filter`/`category_filter` default to this project's actual
    scope (California, FOODS) but are parameters, not hardcoded, so a
    caller can validate the reshape logic itself against a smaller or
    different slice without silently inheriting an assumption baked into
    the function body. Pass None for either to keep all states/categories.

    Validation happens LAST, after filtering - the schema's `unique=["id",
    "date"]` check and dtype checks are cheapest and most meaningful to
    run against the actual data this pipeline will use, not the full
    unfiltered raw file.
    """
    logger.info("reading calendar from %s", calendar_path)
    calendar = read_calendar(calendar_path)

    logger.info("reading and melting sales from %s", sales_path)
    sales_wide = pd.read_csv(sales_path)

    if state_filter is not None:
        sales_wide = sales_wide[sales_wide["state_id"].isin(state_filter)]
    if category_filter is not None:
        sales_wide = sales_wide[sales_wide["cat_id"].isin(category_filter)]

    sales_long = _melt_wide_sales(sales_wide)

    bronze = sales_long.merge(
        calendar[
            [
                "d",
                "date",
                "wm_yr_wk",
                "wday",
                "month",
                "year",
                "event_name_1",
                "event_type_1",
                "event_name_2",
                "event_type_2",
                "snap_CA",
                "snap_TX",
                "snap_WI",
            ]
        ],
        on="d",
        how="left",
    )

    if bronze["date"].isna().any():
        n_missing = int(bronze["date"].isna().sum())
        raise ValueError(
            f"{n_missing} rows failed to match a calendar date after merging on 'd' - "
            f"sales_train_evaluation.csv references a 'd' value not present in calendar.csv"
        )

    bronze["sales"] = bronze["sales"].astype("int64")

    # Same memory lesson as the id columns before melting (see
    # _melt_wide_sales's docstring): these columns each repeat a small
    # set of distinct string values once per row (~11.2M rows on this
    # project's real CA/FOODS data), and the event/type columns are
    # ~99% NaN in the first place. Every downstream feature-building
    # stage does its own defensive `.copy()` of the full bronze table
    # (see build_silver_features), so this table's per-row footprint is
    # paid for repeatedly, not once - shrinking it here, at the one
    # place it's built, benefits every one of those copies at no cost
    # to correctness (category dtype round-trips through pandera's
    # nullable string check the same as object dtype does).
    for col in ["d", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        bronze[col] = bronze[col].astype("category")
    # wm_yr_wk/wday/month/year/snap_* arrive as float64/int64 from the
    # merge; their real ranges (checked by BronzeSalesSchema itself, e.g.
    # wday in [1,7], snap_* in {0,1}) fit comfortably in much smaller
    # integer types.
    bronze["wm_yr_wk"] = bronze["wm_yr_wk"].astype("int32")
    bronze["wday"] = bronze["wday"].astype("int8")
    bronze["month"] = bronze["month"].astype("int8")
    bronze["year"] = bronze["year"].astype("int16")
    for col in ["snap_CA", "snap_TX", "snap_WI"]:
        bronze[col] = bronze[col].astype("int8")

    logger.info(
        "bronze table: %d rows, %d series, %s to %s",
        len(bronze),
        bronze["id"].nunique(),
        bronze["date"].min().date(),
        bronze["date"].max().date(),
    )

    return BronzeSalesSchema.validate(bronze)
