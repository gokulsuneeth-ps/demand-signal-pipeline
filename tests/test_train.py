"""Tests for lightgbm_forecast, the categorical-encoding-consistency fix,
and train_and_log's MLflow-backed run_backtest integration.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from dsp.models.train import (
    CATEGORICAL_COLS,
    _build_feature_matrix,
    build_categorical_dtypes,
    lightgbm_forecast,
    make_lightgbm_forecast_fn,
    train_and_log,
)

warnings.filterwarnings("ignore")


def _lightgbm_native_available() -> tuple[bool, Exception | None]:
    """Checks once whether LightGBM's native library can actually run a
    trivial fit on this machine.

    This is a TEST-ONLY concern, not something production code
    (train.py) needs to work around: unlike ETS's environment fallback
    in baselines.py (a real, designed degradation path when statsforecast
    is unavailable), there is no reasonable "fall back to X" story if
    LightGBM itself can't run at all - that would just mean day 5's
    actual deliverable doesn't work, which is a different situation from
    ETS gracefully stepping down to seasonal-naive.

    Exists because of a real failure caught on a Windows machine: LightGBM
    crashed with `OSError: exception: access violation reading
    0x0000000000000000` from deep inside a ctypes call into its compiled
    lib_lightgbm.dll - reproduced even on a bare, trivial fit with no
    categoricals, no Tweedie, nothing from this project's code involved.
    ctypes translates that Windows SEH access violation into a catchable
    Python OSError rather than a hard interpreter crash, which is what
    makes a graceful pytest skip possible at all here. Likely the same
    class of issue as the Application Control policy that blocked scipy's
    cython_blas DLL in day 4 - an environment constraint, not something
    fixable by reinstalling the package. GitHub Actions' Linux runner is
    unaffected by this Windows-specific failure mode and remains the real
    source of truth for whether these tests pass.
    """
    global _lightgbm_checked, _lightgbm_error
    if not _lightgbm_checked:
        try:
            import lightgbm as lgb

            m = lgb.LGBMRegressor(n_estimators=2, verbose=-1)
            m.fit(np.random.default_rng(0).random((10, 2)), np.random.default_rng(0).random(10))
        except OSError as e:
            _lightgbm_error = e
        _lightgbm_checked = True
    return _lightgbm_error is None, _lightgbm_error


_lightgbm_checked = False
_lightgbm_error: Exception | None = None
_LIGHTGBM_AVAILABLE, _LIGHTGBM_ERROR = _lightgbm_native_available()
requires_lightgbm_native = pytest.mark.skipif(
    not _LIGHTGBM_AVAILABLE,
    reason=f"LightGBM's native library is not usable in this environment: {_LIGHTGBM_ERROR}",
)


def _synthetic_silver_df(n_days: int = 60) -> pd.DataFrame:
    """A tiny dataset shaped like day 3's silver features table - three
    series with distinct base sales levels and a weekly seasonal wiggle,
    enough history for lag_28/rollmean_28 to have real (non-null) values
    for at least the last few days of each series.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2016-01-01", periods=n_days)
    rows = []
    for item, store, base in [
        ("FOODS_1", "CA_1", 10),
        ("FOODS_2", "CA_1", 20),
        ("FOODS_1", "CA_2", 5),
    ]:
        sid = f"{item}_{store}"
        for i, d in enumerate(dates):
            seasonal = 3 * np.sin(2 * np.pi * i / 7)
            sales = max(0.0, base + seasonal + rng.normal(0, 0.5))
            rows.append(
                {
                    "id": sid,
                    "item_id": item,
                    "dept_id": "FOODS_1_DEPT",
                    "store_id": store,
                    "state_id": "CA",
                    "cat_id": "FOODS",
                    "date": d,
                    "d": f"d_{i + 1}",
                    "wm_yr_wk": 11101 + i // 7,
                    "wday": d.dayofweek,
                    "month": d.month,
                    "year": d.year,
                    "sales": sales,
                    "sell_price": 3.98,
                    "event_name_1": None,
                    "event_type_1": None,
                    "event_name_2": None,
                    "event_type_2": None,
                    "snap_CA": int(i % 3 == 0),
                    "snap_TX": 0,
                    "snap_WI": 0,
                    "is_weekend": int(d.dayofweek >= 5),
                    "is_event": 0,
                    "is_snap_day": int(i % 3 == 0),
                    "sales_lag_7": (
                        max(0.0, base + 3 * np.sin(2 * np.pi * (i - 7) / 7)) if i >= 7 else np.nan
                    ),
                    "sales_lag_14": (
                        max(0.0, base + 3 * np.sin(2 * np.pi * (i - 14) / 7)) if i >= 14 else np.nan
                    ),
                    "sales_lag_28": (
                        max(0.0, base + 3 * np.sin(2 * np.pi * (i - 28) / 7)) if i >= 28 else np.nan
                    ),
                    "sales_rollmean_7": base if i >= 7 else np.nan,
                    "sales_rollmean_28": base if i >= 28 else np.nan,
                    "sales_rollstd_7": 1.5 if i >= 7 else np.nan,
                    "sales_rollstd_28": 1.5 if i >= 28 else np.nan,
                    "is_discounted": 0,
                    "price_change": 0.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def silver_df() -> pd.DataFrame:
    return _synthetic_silver_df()


# --- categorical encoding consistency ---


def test_categorical_dtypes_are_identical_across_frames_with_different_subsets(silver_df):
    """Regression-shaped test for the exact LightGBM footgun
    build_categorical_dtypes exists to close: if categories were inferred
    separately per frame, a train_df containing only FOODS_1/FOODS_2 and
    a test_df containing a different subset of ids could silently encode
    the same item_id to different integer codes. Threading one shared
    dtype mapping through both must keep the category ORDER (not just
    membership) identical regardless of which values are actually present
    in either frame.
    """
    categorical_dtypes = build_categorical_dtypes(silver_df, CATEGORICAL_COLS)

    train_df = silver_df[silver_df["id"] == "FOODS_1_CA_1"].iloc[:20]
    test_df = silver_df[silver_df["id"] == "FOODS_2_CA_1"].iloc[20:24]

    X_train, _ = _build_feature_matrix(train_df, categorical_dtypes)
    X_test, _ = _build_feature_matrix(test_df, categorical_dtypes)

    assert list(X_train["item_id"].cat.categories) == list(X_test["item_id"].cat.categories)
    assert list(X_train["store_id"].cat.categories) == list(X_test["store_id"].cat.categories)


def test_build_feature_matrix_excludes_target_and_identifiers(silver_df):
    categorical_dtypes = build_categorical_dtypes(silver_df, CATEGORICAL_COLS)
    X, categorical_cols = _build_feature_matrix(silver_df, categorical_dtypes)

    for excluded in ["id", "date", "d", "wm_yr_wk", "sales", "cat_id", "event_name_1"]:
        assert excluded not in X.columns

    assert set(categorical_cols) == set(CATEGORICAL_COLS)
    for col in categorical_cols:
        assert X[col].dtype.name == "category"


# --- lightgbm_forecast ---


@requires_lightgbm_native
def test_lightgbm_forecast_produces_valid_predictions(silver_df):
    categorical_dtypes = build_categorical_dtypes(silver_df, CATEGORICAL_COLS)
    train_df = silver_df[silver_df["date"] < "2016-02-26"]
    test_df = silver_df[(silver_df["date"] >= "2016-02-26") & (silver_df["date"] < "2016-03-01")]

    preds = lightgbm_forecast(train_df, test_df, categorical_dtypes)

    assert len(preds) == len(test_df)
    assert not preds["prediction"].isna().any()
    assert (preds["prediction"] >= 0).all()  # non-negative floor is enforced
    assert set(preds["id"].unique()) == set(test_df["id"].unique())


@requires_lightgbm_native
def test_lightgbm_forecast_distinguishes_series_by_base_level(silver_df):
    """A weak sanity check that the categorical split is doing SOMETHING
    useful: predictions for the base=20 series should land meaningfully
    higher than predictions for the base=5 series, not be interchangeable.
    """
    categorical_dtypes = build_categorical_dtypes(silver_df, CATEGORICAL_COLS)
    train_df = silver_df[silver_df["date"] < "2016-02-26"]
    test_df = silver_df[(silver_df["date"] >= "2016-02-26") & (silver_df["date"] < "2016-03-01")]

    preds = lightgbm_forecast(train_df, test_df, categorical_dtypes)
    high_base_mean = preds[preds["id"] == "FOODS_2_CA_1"]["prediction"].mean()
    low_base_mean = preds[preds["id"] == "FOODS_1_CA_2"]["prediction"].mean()
    assert high_base_mean > low_base_mean


@requires_lightgbm_native
def test_make_lightgbm_forecast_fn_matches_forecast_fn_contract(silver_df):
    """The whole point of the closure: it must be callable as a plain
    (train_df, test_df) -> predictions_df function, matching run_backtest's
    ForecastFn contract with no extra arguments at the call site.
    """
    categorical_dtypes = build_categorical_dtypes(silver_df, CATEGORICAL_COLS)
    forecast_fn = make_lightgbm_forecast_fn(categorical_dtypes)

    train_df = silver_df[silver_df["date"] < "2016-02-26"]
    test_df = silver_df[(silver_df["date"] >= "2016-02-26") & (silver_df["date"] < "2016-03-01")]
    preds = forecast_fn(train_df, test_df)
    assert len(preds) == len(test_df)


# --- train_and_log / MLflow integration ---


@requires_lightgbm_native
def test_train_and_log_runs_backtest_and_logs_to_mlflow(silver_df, tmp_path, monkeypatch):
    mlflow_db = tmp_path / "mlflow_test.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{mlflow_db}")

    comparison, results = train_and_log(
        silver_df, n_folds=3, horizon=4, min_train_days=14, experiment_name="test-experiment"
    )

    assert len(results["candidate"].per_fold) == 3
    assert len(results["baseline"].per_fold) == 3
    assert len(comparison.fold_deltas) == 3
    assert mlflow_db.exists()

    import sqlite3

    conn = sqlite3.connect(mlflow_db)
    try:
        run_count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert run_count == 1
        metric_keys = {row[0] for row in conn.execute("SELECT DISTINCT key FROM metrics")}
        assert "pooled_wape" in metric_keys
        assert "passed_every_fold" in metric_keys
        param_keys = {row[0] for row in conn.execute("SELECT DISTINCT key FROM params")}
        assert "objective" in param_keys
    finally:
        conn.close()
