# ADR-0001: Gradient-boosted trees over deep learning for the forecasting core

Status: accepted
Date: 2026-08-30

## Context

Demand forecasting can reasonably be approached with statistical models (ETS/ARIMA), tree-based ML
(LightGBM/XGBoost), or deep learning (LSTM/Transformer-based). Time budget is roughly two focused
weeks solo, with a stronger software-engineering than ML-research background.

## Options considered

1. **LSTM / deep learning** — fashionable, matches what a lot of demand-forecasting marketing
   content emphasizes, but expensive to tune well, harder to explain a specific prediction, and
   the top-performing M5-competition solutions were predominantly tree-based, not deep learning.
2. **Pure statistical (ETS/ARIMA)** — fast, well-understood, but doesn't use cross-series signal
   (price, promo, calendar) well, and M5-scale data has enough series for ML to actually help.
3. **Gradient-boosted trees (LightGBM), statistical baselines as the bar to beat** — tabular,
   feature-based, fast to train and explain, matches what most production retail forecasting teams
   actually run.

## Decision

LightGBM as the primary model, with seasonal-naive and ETS as baselines the model must beat on
every backtest fold, not just on average.

## Consequences

Easier: faster iteration, feature importances are directly interpretable, no GPU dependency, plays
to existing engineering strengths. Harder: less impressive-sounding in a one-line pitch than "deep
learning." Ruled out for this build: sequence-to-sequence architectures, attention-based forecasting
— noted as a legitimate stretch direction if the base model is solid and time remains.
