"""Combines bronze + every feature module into the final silver feature
table - the single function `dsp.orchestration.assets` calls for the
"features" pipeline stage, and the same shape/columns every model in
`dsp.models` already depends on (see `dsp.models.train.EXCLUDED_COLS` /
`CATEGORICAL_COLS` for the exact columns downstream code expects).

Takes bronze ALONE, not bronze + a separate prices table - `sell_price`
is already merged into bronze by ingestion (`dsp.ingestion.load.
enrich_with_prices`, run before `BronzeSalesSchema.validate`), so this
module's job starts after that's already true. See `prices.py`'s module
docstring for why a second, independent price merge used to live here and
was removed - it was a latent bug (a second merge colliding with the
`sell_price` column ingestion already produced), not a feature.
"""

from __future__ import annotations

import gc
import logging

import pandas as pd

from dsp.features.calendar import add_calendar_features
from dsp.features.lags import add_lag_features, add_rolling_features
from dsp.features.prices import add_price_features

logger = logging.getLogger(__name__)


def build_silver_features(bronze_df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full bronze -> silver feature pipeline. Calendar features
    are applied first simply because they're cheapest and have no
    dependency on anything else in this function; price features need
    `sell_price` (already on `bronze_df` - see module docstring), and lag/
    rolling sales features only need `sales` + `id` + `date`, so neither
    has an ordering dependency on the other.

    Explicitly `del`s each stage's input immediately after producing the
    next stage's output, followed by `gc.collect()`. This isn't
    stylistic: every `add_*` function below does its own defensive
    `df.copy()` (so callers never have a function silently mutate a frame
    out from under them), and on this project's real CA/FOODS data
    (~11.2M rows) each copy is ~2-3GB. Without explicitly dropping the
    superseded reference, Python's local-variable scoping keeps the OLD
    frame alive for the rest of this function's body even though it's
    never read again (a local name isn't freed just because it's unused -
    only when reassigned, deleted, or the function returns), so every
    stage would hold two full copies at once and stack on top of the
    previous stage's now-orphaned memory. That is exactly what OOM-killed
    this pipeline the first time it ran against the real raw data: RSS
    climbed stage over stage (~2GB -> ~4.4GB and rising) instead of
    holding steady at roughly one frame's worth. Freeing each predecessor
    the moment it's superseded keeps peak memory bounded to ~2 live
    copies (old + new) during a single stage's `.copy()`, not N stages'
    worth stacked on top of each other.
    """
    logger.info("building silver features from %d bronze rows", len(bronze_df))

    out = add_calendar_features(bronze_df)
    del bronze_df
    gc.collect()

    out2 = add_price_features(out)
    del out
    gc.collect()

    out3 = add_lag_features(out2)
    del out2
    gc.collect()

    out4 = add_rolling_features(out3)
    del out3
    gc.collect()

    logger.info("silver feature table: %d rows, %d columns", len(out4), out4.shape[1])
    return out4


def build_silver_features_chunked(
    bronze_df: pd.DataFrame, chunk_col: str = "store_id"
) -> pd.DataFrame:
    """Same output as `build_silver_features`, computed one `chunk_col`
    value at a time and concatenated, to bound peak memory on large real
    data instead of holding several full-table copies at once (see
    `build_silver_features`'s docstring for why those copies happen).

    `chunk_col` defaults to `store_id`, not an arbitrary row slice,
    because it's a partition every feature function here already
    respects: `id` (this project's per-series lag/rolling group key) is
    itself a store+item combination, so no series' lag or rolling window
    ever spans two stores - chunking on `store_id` changes nothing about
    which rows a given feature is computed from, only the order/batching
    of the computation. Chunking on an arbitrary row range instead would
    risk silently splitting a single series' date history across chunks
    and corrupting its lag/rolling values; this function would rather
    raise than guess, so it checks that no `id` value spans two chunks.

    On this project's real CA/FOODS data (4 stores, ~1,437 series each,
    ~11.2M total rows) this keeps peak RSS to roughly a quarter of the
    single-shot version - confirmed by direct measurement, not assumed -
    while producing byte-identical output (verified in
    tests/test_build_features.py by comparing both paths against the
    same input).
    """
    if chunk_col not in bronze_df.columns:
        raise ValueError(
            f"build_silver_features_chunked: chunk_col {chunk_col!r} not in bronze_df columns"
        )

    # Enforced, not assumed: a series (`id`) that spanned two chunk_col
    # values would have its lag/rolling history silently split across
    # two independently-processed chunks, corrupting those features for
    # every row near the split with no error at all. Checking this costs
    # one groupby over columns already in memory - cheap relative to the
    # risk of a silent correctness bug in exactly the feature class this
    # project has been most careful about (see lags.py's module
    # docstring on leakage).
    ids_per_chunk_value = bronze_df.groupby("id", observed=True)[chunk_col].nunique()
    bad_ids = ids_per_chunk_value[ids_per_chunk_value > 1]
    if len(bad_ids) > 0:
        raise ValueError(
            f"build_silver_features_chunked: {len(bad_ids)} id value(s) span more than one "
            f"{chunk_col} value (e.g. {bad_ids.index[0]!r}) - chunking by {chunk_col!r} would "
            f"silently split that series' lag/rolling history across chunks. Choose a chunk_col "
            f"that never varies within a single id, or don't chunk."
        )

    chunk_values = sorted(bronze_df[chunk_col].astype(str).unique())
    logger.info("building silver features in %d chunks by %s", len(chunk_values), chunk_col)

    results = []
    for value in chunk_values:
        bronze_chunk = bronze_df[bronze_df[chunk_col].astype(str) == value].copy()
        # Filtering rows out of a `category`-dtype column does NOT shrink
        # its category list - `bronze_chunk["id"]` still carries all
        # 5,748 original categories even though only ~1,437 have any
        # rows in this chunk. Left alone, that's not just extra memory:
        # a categorical groupby without `observed=True` iterates every
        # category including the now-empty ones, and this project's own
        # rolling-window feature code crashed on exactly that (an empty
        # group's window indexer). The feature functions now pass
        # `observed=True` everywhere they group by id_col, which is the
        # real fix; this is the defensive second half of it, so no other
        # category-dtype-aware code downstream (or added later) inherits
        # a silently stale category list for this chunk.
        for col in bronze_chunk.select_dtypes(include="category").columns:
            bronze_chunk[col] = bronze_chunk[col].cat.remove_unused_categories()

        silver_chunk = build_silver_features(bronze_chunk)
        results.append(silver_chunk)
        del bronze_chunk, silver_chunk
        gc.collect()
        logger.info("chunk %s done (%d/%d)", value, len(results), len(chunk_values))

    del bronze_df
    gc.collect()

    out = pd.concat(results, ignore_index=True)
    del results
    gc.collect()
    return out
