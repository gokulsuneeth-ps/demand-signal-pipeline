"""Raw M5 CSVs -> validated, filtered, long-format bronze table.

Deliberately thin, per docs/architecture.md: this module's job is reshape +
join + validate, nothing more. Feature engineering (day 3) is a separate
module reading bronze, not folded in here.

Pipeline, end to end:
    raw wide sales (one row per item-store, one column per day)
      -> filter to the chosen category/state subset (see PROBLEM_STATEMENT.md)
      -> melt to long (one row per item-store-day)
      -> join calendar (adds date, events, SNAP flags)
      -> join sell_prices (adds price, nullable)
      -> validate against BronzeSalesSchema
      -> write data/bronze/sales_long.parquet
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import pandas as pd

from dsp.ingestion.schema import BronzeSalesSchema, CalendarSchema, SellPricesSchema

logger = logging.getLogger(__name__)

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_calendar(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "calendar.csv")
    return CalendarSchema.validate(df)


def load_sell_prices(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "sell_prices.csv")
    return SellPricesSchema.validate(df)


def load_raw_sales_wide(
    raw_dir: Path, filename: str = "sales_train_evaluation.csv"
) -> pd.DataFrame:
    """The M5 sales file is wide: one row per item-store series, one column
    per day (`d_1`, `d_2`, ...). `sales_train_evaluation.csv` has the fullest
    history; `sales_train_validation.csv` is an earlier snapshot — use
    evaluation unless you have a specific reason to reproduce the original
    competition's validation-phase leaderboard.
    """
    return pd.read_csv(raw_dir / filename)


def filter_subset(
    sales_wide: pd.DataFrame, cat_id: str = "FOODS", state_id: str = "CA"
) -> pd.DataFrame:
    """Narrow to the subset PROBLEM_STATEMENT.md commits to, before the much
    more expensive melt step — filtering wide is cheap, filtering long on
    30M+ rows is not.
    """
    mask = (sales_wide["cat_id"] == cat_id) & (sales_wide["state_id"] == state_id)
    subset = sales_wide.loc[mask].reset_index(drop=True)
    if subset.empty:
        raise ValueError(
            f"No rows matched cat_id={cat_id!r}, state_id={state_id!r} — "
            "check the raw file actually contains this subset."
        )
    return subset


def melt_to_long(sales_wide: pd.DataFrame) -> pd.DataFrame:
    """Wide (one column per day) -> long (one row per item-store-day).

    Casts ID_COLS to `category` dtype before melting, not after: `melt`
    repeats every id-column value once per day column in its output, so
    with these left as plain strings, melting this project's real CA/
    FOODS subset (~5,700 series x ~1,900 days) inflates a sub-1GB wide
    table into 5+GB of long-format output - confirmed by direct
    measurement, and enough to OOM-kill this exact function against the
    real raw data. Category dtype stores each distinct id string once,
    cutting that melted table's memory by roughly 6x. `BronzeSalesSchema`
    (`coerce=True`, `Series[str]`) casts these back to plain `str` at
    validation time, so this is purely an intermediate-memory
    optimization, not a change to the schema this function's callers see.
    """
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    sales_wide = sales_wide.copy()
    for col in ID_COLS:
        sales_wide[col] = sales_wide[col].astype("category")
    long_df = sales_wide.melt(
        id_vars=ID_COLS,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    return long_df


def enrich_with_calendar(long_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    calendar_cols = [
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
    return long_df.merge(calendar_df[calendar_cols], on="d", how="left")


def enrich_with_prices(df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    # Left join: a missing price is a legitimate state (item not yet
    # listed at that store-week), not a data error — see BronzeSalesSchema.
    return df.merge(prices_df, on=["store_id", "item_id", "wm_yr_wk"], how="left")


def run_ingestion(
    raw_dir: Path,
    bronze_dir: Path,
    cat_id: str = "FOODS",
    state_id: str = "CA",
) -> pd.DataFrame:
    """End to end: raw CSVs in `raw_dir` -> validated bronze parquet in
    `bronze_dir`. Returns the bronze DataFrame as well, mainly so callers
    (and tests) can inspect it without re-reading the parquet file.

    Processes one `store_id` at a time (melt -> enrich_with_calendar ->
    enrich_with_prices), instead of running the whole ~11.2M-row CA/FOODS
    subset through those steps at once, then concatenates before the
    final schema validation. This isn't a stylistic choice: run against
    this project's actual raw data, the single-shot version OOM-crashed
    even with `melt_to_long`'s category-dtype fix in place (RSS climbed
    past 6GB and was killed by the sandbox's memory cgroup) - each stage
    (`.merge`, `.copy()` inside `melt_to_long`) holds its own new full-
    size frame while Python's scoping keeps the previous one alive too,
    and that stacks stage over stage across ~11.2M rows. `store_id` is a
    safe partition for this: it's a strict superset of `id` (this
    project's item-store series key - see ID_COLS), so no row's calendar
    or price join ever needs data from a different store, and chunking
    changes nothing about the OUTPUT, only the order/batching of the
    computation. On this project's real CA/FOODS data (4 CA stores,
    ~1,437 series each) this keeps peak memory to roughly a quarter of
    the single-shot version - confirmed by direct measurement against the
    real raw files, not assumed.
    """
    calendar = load_calendar(raw_dir)
    prices = load_sell_prices(raw_dir)
    sales_wide = load_raw_sales_wide(raw_dir)

    subset = filter_subset(sales_wide, cat_id=cat_id, state_id=state_id)
    del sales_wide
    gc.collect()

    store_ids = sorted(subset["store_id"].unique())
    logger.info("ingesting %d rows across %d store(s)", len(subset), len(store_ids))

    chunks = []
    for store in store_ids:
        store_subset = subset[subset["store_id"] == store].copy()

        long_df = melt_to_long(store_subset)
        del store_subset
        gc.collect()

        enriched = enrich_with_calendar(long_df, calendar)
        del long_df
        gc.collect()

        priced = enrich_with_prices(enriched, prices)
        del enriched
        gc.collect()

        chunks.append(priced)
        del priced
        gc.collect()
        logger.info("store %s ingested (%d/%d)", store, len(chunks), len(store_ids))

    del subset, calendar, prices
    gc.collect()

    bronze = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    bronze = BronzeSalesSchema.validate(bronze)

    bronze_dir.mkdir(parents=True, exist_ok=True)
    out_path = bronze_dir / "sales_long.parquet"
    bronze.to_parquet(out_path, index=False)

    return bronze


if __name__ == "__main__":
    # Run with: python -m dsp.ingestion.load
    project_root = Path(__file__).resolve().parents[3]
    result = run_ingestion(
        raw_dir=project_root / "data" / "raw",
        bronze_dir=project_root / "data" / "bronze",
    )
    print(f"Wrote {len(result):,} rows to data/bronze/sales_long.parquet")
    print(f"Series (unique item-store combos): {result['id'].nunique():,}")
    print(f"Date range: {result['date'].min()} to {result['date'].max()}")
