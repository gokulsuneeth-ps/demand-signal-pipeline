"""Orchestrates bronze -> feature-engineered silver table.

Deliberately just a sequence of calls to the pure functions in
calendar.py, lags.py, and prices.py — this module owns wiring, not logic,
so a bug is always traceable to one specific, individually-tested
function rather than to "somewhere in build_features."
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dsp.features.calendar import add_calendar_features
from dsp.features.lags import add_lag_features, add_rolling_features
from dsp.features.prices import add_price_features


def build_features(bronze_df: pd.DataFrame) -> pd.DataFrame:
    """bronze (validated long-format sales) -> silver (feature-engineered).

    Row count in equals row count out — features are added as columns,
    no rows are dropped here. The early rows of each series will have
    null lag/rolling features (there's no 28 days of history before day
    1) — left as nulls rather than dropped or zero-filled, since LightGBM
    (day 5) splits on missingness natively, and silently zero-filling
    would tell the model "sales were 0" when the truth is "unknown,"
    which is a meaningfully different and worse signal.
    """
    out = add_calendar_features(bronze_df)
    out = add_lag_features(out)
    out = add_rolling_features(out)
    out = add_price_features(out)
    return out


def run_feature_build(bronze_path: Path, silver_dir: Path) -> pd.DataFrame:
    bronze_df = pd.read_parquet(bronze_path)
    silver_df = build_features(bronze_df)

    silver_dir.mkdir(parents=True, exist_ok=True)
    out_path = silver_dir / "features.parquet"
    silver_df.to_parquet(out_path, index=False)

    return silver_df


if __name__ == "__main__":
    # Run with: python -m dsp.features.build
    project_root = Path(__file__).resolve().parents[3]
    result = run_feature_build(
        bronze_path=project_root / "data" / "bronze" / "sales_long.parquet",
        silver_dir=project_root / "data" / "silver",
    )
    feature_cols = [
        c
        for c in result.columns
        if c
        not in {
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "d",
            "sales",
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
            "sell_price",
        }
    ]
    n_new = len(feature_cols)
    print(f"Wrote {len(result):,} rows, {n_new} new feature cols -> data/silver/features.parquet")
    print(f"New columns: {feature_cols}")
    null_pct = result[feature_cols].isna().mean().round(3)
    print("Null fraction per feature (nonzero expected for lag/rolling early rows):")
    print(null_pct)
