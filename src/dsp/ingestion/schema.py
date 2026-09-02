"""Pandera schemas for the raw -> bronze validation step.

These exist to catch a bad raw download or a reshape bug LOUDLY, at
ingestion time, rather than as a confusing downstream failure three
pipeline stages later - the same "raise, don't silently degrade"
discipline this project has applied everywhere else (metrics.py,
backtest.py, baselines.py).

Two schemas: `CalendarSchema` checks the raw calendar.csv exactly as
downloaded; `BronzeSalesSchema` checks the RESHAPED long-format sales
table `ingest.build_bronze` produces (one row per series per day), not
the raw wide-format CSV (1,941 day columns is not something pandera's
column-based checks are a good fit for - the wide file's own structural
sanity, e.g. that every d_1..d_1941 column exists and is numeric, is
checked directly in `ingest.py` instead, where the melt happens).
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

CalendarSchema = DataFrameSchema(
    {
        "date": Column(pa.DateTime),
        "wm_yr_wk": Column(pa.Int, checks=pa.Check.ge(0)),
        "wday": Column(pa.Int, checks=pa.Check.in_range(1, 7)),
        "month": Column(pa.Int, checks=pa.Check.in_range(1, 12)),
        "year": Column(pa.Int, checks=pa.Check.ge(2010)),
        "d": Column(pa.String),
        "event_name_1": Column(pa.String, nullable=True),
        "event_type_1": Column(pa.String, nullable=True),
        "event_name_2": Column(pa.String, nullable=True),
        "event_type_2": Column(pa.String, nullable=True),
        "snap_CA": Column(pa.Int, checks=pa.Check.isin([0, 1])),
        "snap_TX": Column(pa.Int, checks=pa.Check.isin([0, 1])),
        "snap_WI": Column(pa.Int, checks=pa.Check.isin([0, 1])),
    },
    strict=False,  # calendar.csv also has a "weekday" text column this project doesn't use
    coerce=False,
)

BronzeSalesSchema = DataFrameSchema(
    {
        # These five id-ish columns are `category` dtype, not plain
        # string, by deliberate design - see build_bronze's docstring
        # in ingest.py: melting the wide raw file with these left as
        # plain strings repeats every value once per day column and
        # inflates the melted table roughly 6x, enough to OOM-kill the
        # process on this project's actual sandbox. pd.CategoricalDtype()
        # here accepts any category set (unconstrained), matching the
        # role of these columns as identifiers, not a validated
        # enumeration.
        "id": Column(pd.CategoricalDtype()),
        "item_id": Column(pd.CategoricalDtype()),
        "dept_id": Column(pd.CategoricalDtype()),
        "cat_id": Column(pd.CategoricalDtype()),
        "store_id": Column(pd.CategoricalDtype()),
        "state_id": Column(pd.CategoricalDtype()),
        "date": Column(pa.DateTime),
        # `d`, the two event-name and two event-type columns are
        # `category` dtype, not plain string, for the same memory reason
        # as the id columns above: see build_bronze's docstring in
        # ingest.py. pd.CategoricalDtype() is unconstrained (any category
        # set), and pandera's nullable check still applies on top of it.
        "d": Column(pd.CategoricalDtype()),
        "wm_yr_wk": Column(pa.Int32, checks=pa.Check.ge(0)),
        "wday": Column(pa.Int8, checks=pa.Check.in_range(1, 7)),
        "month": Column(pa.Int8, checks=pa.Check.in_range(1, 12)),
        "year": Column(pa.Int16, checks=pa.Check.ge(2010)),
        # Sales can never be negative - a real data-quality invariant, not
        # a stylistic preference. A raw file that violates this should
        # fail ingestion loudly, not silently propagate a negative
        # "demand" number into every downstream metric and model.
        "sales": Column(pa.Int, checks=pa.Check.ge(0)),
        "event_name_1": Column(pd.CategoricalDtype(), nullable=True),
        "event_type_1": Column(pd.CategoricalDtype(), nullable=True),
        "event_name_2": Column(pd.CategoricalDtype(), nullable=True),
        "event_type_2": Column(pd.CategoricalDtype(), nullable=True),
        "snap_CA": Column(pa.Int8, checks=pa.Check.isin([0, 1])),
        "snap_TX": Column(pa.Int8, checks=pa.Check.isin([0, 1])),
        "snap_WI": Column(pa.Int8, checks=pa.Check.isin([0, 1])),
    },
    # No duplicate (id, date) rows - the single most important structural
    # invariant of a "long format, one row per series per day" table.
    # Every downstream stage (lag features, folds, backtest) silently
    # assumes this holds; checking it once here means a reshape bug gets
    # caught at ingestion, not as a mysterious doubled WAPE denominator
    # three stages later.
    unique=["id", "date"],
    strict=True,
    coerce=False,
)
