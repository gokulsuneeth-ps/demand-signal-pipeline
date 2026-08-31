"""Calendar/event features.

Pure functions: DataFrame in, DataFrame out, no side effects. Each is
individually unit-tested rather than only checked as part of the full
pipeline, so a bug here shows up as one failing test pointing at one
function, not a confusing failure three stages downstream.
"""

from __future__ import annotations

import pandas as pd

# Maps a series' state to the correct SNAP-benefit column. The current
# subset (PROBLEM_STATEMENT.md) is CA-only, so only snap_CA is ever
# selected in practice today — but writing this as a lookup rather than
# hardcoding snap_CA means adding TX/WI later is a data-scope change, not
# a code change.
_SNAP_COLUMN_BY_STATE = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}


def add_weekend_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `is_weekend` from the actual calendar date, not M5's `wday`
    column — M5's wday numbering (1=Saturday) is easy to misread as
    Python's `.dayofweek` (0=Monday), so deriving it fresh from `date`
    avoids a whole class of off-by-one bugs.
    """
    out = df.copy()
    dow = pd.to_datetime(out["date"]).dt.dayofweek  # 0=Mon ... 6=Sun
    out["is_weekend"] = dow.isin([5, 6]).astype(int)
    return out


def add_event_flag(df: pd.DataFrame) -> pd.DataFrame:
    """`is_event`: 1 if either of M5's two event slots is populated.
    Deliberately collapsed to a single binary flag rather than one-hot
    encoding every event name — with this subset's row count, a rare
    event name would be a near-constant column that helps LightGBM very
    little and mostly adds noise. Event *type* (Sporting/Cultural/etc.)
    is left as a stretch feature, not built here.
    """
    out = df.copy()
    out["is_event"] = (out["event_name_1"].notna() | out["event_name_2"].notna()).astype(int)
    return out


def add_snap_flag(df: pd.DataFrame) -> pd.DataFrame:
    """`is_snap_day`: whether SNAP (food-assistance) benefits are usable
    that day in the series' own state — a real demand driver for a FOODS
    category specifically, unlike a generic "any state's SNAP day" flag
    would be.
    """
    out = df.copy()
    if out["state_id"].nunique() > 1:
        # Correct but slower per-row lookup, only pays its cost once the
        # subset actually spans multiple states.
        col_per_row = out["state_id"].map(_SNAP_COLUMN_BY_STATE)
        out["is_snap_day"] = [out.loc[i, col] for i, col in col_per_row.items()]
    else:
        state = out["state_id"].iloc[0]
        out["is_snap_day"] = out[_SNAP_COLUMN_BY_STATE[state]]
    out["is_snap_day"] = out["is_snap_day"].astype(int)
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper applying all three calendar features in order."""
    return add_snap_flag(add_event_flag(add_weekend_flag(df)))
