"""LightGBM training entrypoint + MLflow logging.

`lightgbm_forecast` matches the same `(train_df, test_df) -> predictions_df`
contract as `baselines.seasonal_naive_forecast` - it's a real model plugged
into the exact same `run_backtest` harness from day 4, no changes needed
there. `make_lightgbm_forecast_fn` exists only to bind the category
encoding (see `build_categorical_dtypes` below) via closure, since
`run_backtest` calls `forecast_fn(train_df, test_df)` with exactly those
two arguments and nothing else - same adapter pattern the day 4 design
discussion anticipated for `ets_forecast`.

`train_and_log` wraps `run_backtest` with MLflow tracking; `run_backtest`
itself stays completely unaware MLflow exists, matching the day 4
decision to keep `run_backtest`/`compares_favorably` independently
testable and free of any single caller's concerns.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from dsp.models.backtest import ComparisonResult, ForecastFn, compares_favorably, run_backtest
from dsp.models.baselines import seasonal_naive_forecast

logger = logging.getLogger(__name__)

# Columns that must never reach the feature matrix: identifiers/dates, the
# target itself, raw columns superseded by an already-engineered flag
# (event_name_*/event_type_* -> is_event; snap_CA/TX/WI -> is_snap_day),
# and cat_id (constant "FOODS" for this dataset's whole subset - zero
# variance, no signal, just noise in the split search).
EXCLUDED_COLS = {
    "id",
    "date",
    "d",
    "wm_yr_wk",
    "sales",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "cat_id",
}

# Passed to LightGBM as native categoricals (not one-hot encoded - item_id
# alone has ~1,437 levels in the CA/FOODS subset).
CATEGORICAL_COLS = ["item_id", "dept_id", "store_id", "state_id"]

DEFAULT_LGBM_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.2,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "verbose": -1,
}


def build_categorical_dtypes(
    df: pd.DataFrame, categorical_cols: list[str] = CATEGORICAL_COLS
) -> dict[str, pd.CategoricalDtype]:
    """Computes each categorical column's category set ONCE from the full
    dataset, not separately per fold's train/test frames.

    This exists to close a real LightGBM footgun: if a categorical
    column's categories are inferred independently from train_df and
    test_df (pandas' default `astype("category")` behavior), the same
    string value can silently get a DIFFERENT integer category code in
    each frame - LightGBM splits on the integer code, not the string, so
    a model fit on one encoding and asked to predict against another can
    silently produce nonsense with no error. Computing the category set
    once, up front, from the union of all values the model will ever see
    (train AND test, across every fold), and threading that same mapping
    into every fold's fit and predict calls guarantees "FOODS_1_CA_1"
    means the same category code everywhere, always.
    """
    dtypes = {}
    for col in categorical_cols:
        categories = sorted(df[col].dropna().unique().tolist())
        dtypes[col] = pd.CategoricalDtype(categories=categories)
    return dtypes


def _build_feature_matrix(
    df: pd.DataFrame, categorical_dtypes: dict[str, pd.CategoricalDtype]
) -> tuple[pd.DataFrame, list[str]]:
    """Selects feature columns (everything not in EXCLUDED_COLS) and casts
    CATEGORICAL_COLS to the shared dtype from `build_categorical_dtypes`.

    Null values in lag/rolling features are passed through untouched, not
    imputed - LightGBM splits on missingness natively, and day 3's
    build_features already made the case that zero-filling would falsely
    encode "not enough history yet" as "sales were zero," a meaningfully
    different and worse signal.
    """
    feature_cols = [c for c in df.columns if c not in EXCLUDED_COLS]
    X = df[feature_cols].copy()

    categorical_cols_present = [c for c in CATEGORICAL_COLS if c in X.columns]
    for col in categorical_cols_present:
        X[col] = X[col].astype(categorical_dtypes[col])

    return X, categorical_cols_present


def lightgbm_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_dtypes: dict[str, pd.CategoricalDtype],
    params: dict | None = None,
) -> pd.DataFrame:
    """Fits a fresh LightGBM model on train_df and predicts test_df.

    A fresh model per fold, never one model trained once and evaluated
    across multiple "past" windows - training once on all data and
    scoring it against earlier folds would leak each fold's own future
    into other folds' evaluation, the same rolling-origin discipline
    `run_backtest` already enforces for the baselines.

    `categorical_dtypes` must come from `build_categorical_dtypes` run
    once over the FULL dataset (not just this fold) - see that function's
    docstring for why a per-fold-computed encoding is unsafe.

    Predictions are clipped at 0: the tweedie objective's log-link
    normally keeps predictions non-negative on its own, but sales can
    never legitimately be negative, so this is a defensive floor against
    any floating-point edge case rather than a correction the model is
    expected to need.
    """
    resolved_params = dict(params) if params is not None else dict(DEFAULT_LGBM_PARAMS)

    X_train, categorical_cols = _build_feature_matrix(train_df, categorical_dtypes)
    y_train = train_df["sales"]
    X_test, _ = _build_feature_matrix(test_df, categorical_dtypes)

    model = lgb.LGBMRegressor(**resolved_params)
    model.fit(X_train, y_train, categorical_feature=categorical_cols)

    predictions = np.clip(model.predict(X_test), a_min=0, a_max=None)

    return pd.DataFrame(
        {
            "id": test_df["id"].to_numpy(),
            "date": test_df["date"].to_numpy(),
            "prediction": predictions,
        }
    )


def make_lightgbm_forecast_fn(
    categorical_dtypes: dict[str, pd.CategoricalDtype],
    params: dict | None = None,
) -> ForecastFn:
    """Binds categorical_dtypes and params via closure, producing a plain
    `(train_df, test_df) -> predictions_df` callable - `run_backtest`'s
    exact `ForecastFn` contract, with no changes to `run_backtest` itself.
    """

    def _forecast_fn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
        return lightgbm_forecast(train_df, test_df, categorical_dtypes, params)

    return _forecast_fn


def train_and_log(
    df: pd.DataFrame,
    n_folds: int,
    horizon: int,
    min_train_days: int,
    params: dict | None = None,
    experiment_name: str = "dsp-lightgbm",
) -> tuple[ComparisonResult, dict]:
    """Runs the LightGBM backtest AND a seasonal-naive backtest over the
    same folds, compares them, and logs everything to MLflow.

    Returns (comparison, results) where results holds both BacktestResults
    for inspection - PROBLEM_STATEMENT.md's definition of done
    (`comparison.passed_every_fold`) is checked here, not silently assumed.
    """
    mlflow.set_experiment(experiment_name)
    resolved_params = dict(params) if params is not None else dict(DEFAULT_LGBM_PARAMS)

    categorical_dtypes = build_categorical_dtypes(df, CATEGORICAL_COLS)
    forecast_fn = make_lightgbm_forecast_fn(categorical_dtypes, resolved_params)

    with mlflow.start_run():
        mlflow.log_params(resolved_params)
        mlflow.log_params(
            {"n_folds": n_folds, "horizon": horizon, "min_train_days": min_train_days}
        )

        candidate_result = run_backtest(df, forecast_fn, n_folds, horizon, min_train_days)
        baseline_result = run_backtest(
            df, seasonal_naive_forecast, n_folds, horizon, min_train_days
        )
        comparison = compares_favorably(candidate_result, baseline_result)

        for fold_result in candidate_result.per_fold:
            n = fold_result.fold.fold_number
            mlflow.log_metric(f"fold_{n}_wape", fold_result.wape, step=n)
            mlflow.log_metric(f"fold_{n}_rmse", fold_result.rmse, step=n)
            mlflow.log_metric(f"fold_{n}_mape", fold_result.mape, step=n)

        mlflow.log_metric("pooled_wape", candidate_result.pooled_wape)
        mlflow.log_metric("pooled_rmse", candidate_result.pooled_rmse)
        mlflow.log_metric("pooled_mape", candidate_result.pooled_mape)
        mlflow.log_metric("baseline_pooled_wape", baseline_result.pooled_wape)
        mlflow.log_metric("passed_every_fold", int(comparison.passed_every_fold))
        mlflow.log_metric("pooled_wape_delta_vs_baseline", comparison.pooled_delta)

        logger.info(
            "candidate pooled_wape=%.4f baseline pooled_wape=%.4f passed_every_fold=%s",
            candidate_result.pooled_wape,
            baseline_result.pooled_wape,
            comparison.passed_every_fold,
        )

    return comparison, {"candidate": candidate_result, "baseline": baseline_result}
