"""Calendar/event features (day-of-week, holiday flags, SNAP flags).

Each function here is a pure function (DataFrame in, DataFrame out),
trivially unit-testable without spinning up the full pipeline, matching
the day-1 stub's stated intent.
"""

from __future__ import annotations

import pandas as pd

# M5's own convention (confirmed against the real calendar.csv): wday=1 is
# Saturday, wday=2 is Sunday - the M5 week starts Saturday, not Monday.
WEEKEND_WDAY_VALUES = {1, 2}


def add_is_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `is_weekend` from the `wday` column, using M5's own
    Saturday=1/Sunday=2 weekend convention (not Python's Monday=0
    convention) - stated explicitly since silently assuming the wrong
    convention would mislabel every single row without ever raising an
    error.
    """
    out = df.copy()
    out["is_weekend"] = out["wday"].isin(WEEKEND_WDAY_VALUES).astype(int)
    return out


def add_is_event(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `is_event`: 1 if EITHER event_name_1 or event_name_2 is set
    for that day (a day can have two named events, e.g. a religious
    observance overlapping a sporting event) - checking only
    event_name_1 would silently miss every day where only the second
    event slot was populated.
    """
    out = df.copy()
    has_event_1 = out["event_name_1"].notna()
    has_event_2 = out["event_name_2"].notna()
    out["is_event"] = (has_event_1 | has_event_2).astype(int)
    return out


def add_is_snap_day(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `is_snap_day`: whether THIS ROW'S OWN STATE had SNAP
    (food-assistance) benefits active that day.

    This is the one calendar feature that must be read per-row, not from
    a single fixed column: the raw calendar table carries separate
    snap_CA/snap_TX/snap_WI columns, and a row's relevant column depends
    on its OWN `state_id` - a Texas row must read snap_TX, not snap_CA.
    Reading a single hardcoded snap column (e.g. always snap_CA) would
    silently produce the wrong flag for every non-CA row; this project's
    actual scope is CA-only, but this function is written correctly for
    any state mix rather than quietly baking in that scope assumption.

    Raises if `state_id` contains a value with no matching snap_<STATE>
    column, rather than silently leaving is_snap_day undefined (e.g. as
    NaN) for those rows.
    """
    out = df.copy()
    states = out["state_id"].unique()
    missing_cols = [s for s in states if f"snap_{s}" not in out.columns]
    if missing_cols:
        raise ValueError(
            f"add_is_snap_day: no snap_<STATE> column for state_id value(s) "
            f"{missing_cols} - expected one of {[c for c in out.columns if c.startswith('snap_')]}"
        )

    out["is_snap_day"] = 0
    for state in states:
        mask = out["state_id"] == state
        out.loc[mask, "is_snap_day"] = out.loc[mask, f"snap_{state}"].astype(int)
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper applying all three calendar features in the
    project's standard order. Each function above stays independently
    testable and usable on its own; this is just what `features/build.py`
    actually calls.
    """
    out = add_is_weekend(df)
    out = add_is_event(out)
    out = add_is_snap_day(out)
    return out
