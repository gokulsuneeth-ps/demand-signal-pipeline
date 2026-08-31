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

from pathlib import Path

import pandas as pd

from dsp.ingestion.schema import BronzeSalesSchema, CalendarSchema, SellPricesSchema

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
    """Wide (one column per day) -> long (one row per item-store-day)."""
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
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
    """
    calendar = load_calendar(raw_dir)
    prices = load_sell_prices(raw_dir)
    sales_wide = load_raw_sales_wide(raw_dir)

    subset = filter_subset(sales_wide, cat_id=cat_id, state_id=state_id)
    long_df = melt_to_long(subset)
    long_df = enrich_with_calendar(long_df, calendar)
    bronze = enrich_with_prices(long_df, prices)

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
