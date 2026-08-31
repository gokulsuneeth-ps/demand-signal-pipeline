# Problem statement

Written before any modeling starts, so "good" has a definition before there's a number to rationalize.

## Task

Forecast daily unit demand at **item × store** granularity, for a **28-day horizon**, refreshed daily (rolling forecast, not a one-shot annual number).

## Dataset

**M5 Forecasting — Accuracy** (Walmart, via the Makridakis M5 competition, hosted on Kaggle).
Full dataset: ~30,490 item-store series across 10 stores in 3 US states, 3 years of daily history, with calendar events, SNAP-benefit flags, and sell prices.

**Subset used here:** `FOODS` category, California stores only → roughly 1,400–1,800 series. Chosen because:
- Food/grocery is the category with the clearest weekly seasonality and promo sensitivity — the part of the dataset where ML actually has something to learn over a naive seasonal baseline.
- Single-state keeps daily backtesting fast enough to iterate on a laptop (full dataset backtesting is a multi-hour job; this subset should run in minutes).
- Small enough to reason about by hand when debugging — you can pull up ten SKUs and actually look at them.

Full-dataset scaling is a stretch goal (see ROADMAP checklist), not a day-1–10 requirement.

## Forecast horizon & refresh cadence

- Horizon: 28 days ahead (matches the M5 competition's own horizon, so results are comparable to a known benchmark).
- Cadence: daily refresh in principle; for the portfolio build, backtesting simulates this via rolling-origin folds rather than literally running daily for months.

## Success metric — defined before training anything

Primary metric: **WAPE (Weighted Absolute Percentage Error)**, weighted by each series' historical volume. WAPE over per-series MAPE because MAPE blows up on low-volume/intermittent series (common in retail SKUs) and WAPE is the metric actually used in most production retail forecasting teams.

Secondary metrics logged alongside it: RMSE (penalizes large misses, useful for stockout-risk framing) and MAPE (reported for comparability with the article's and most public benchmarks' claims — but not the metric decisions are made on).

**Baseline to beat:** seasonal-naive (repeat the value from 7 days prior — grocery demand's dominant seasonality is weekly). The model is only worth using if it beats seasonal-naive by a statistically meaningful margin on held-out folds, not just on average.

**Definition of done for the modeling stage:** LightGBM model's WAPE is lower than seasonal-naive's WAPE on every rolling-origin fold, not just on average — a model that wins on average but loses on 2 of 5 folds is not "better," it's inconsistent, and that inconsistency belongs in the README's limitations section either way.

## What's explicitly out of scope

- Multi-echelon inventory optimization (this repo has one simplified reorder-point policy, documented as such — not a full MRP/DRP system).
- Real-time/streaming ingestion (batch only — see docs/architecture.md).
- New-product / cold-start forecasting (all series in the chosen subset have full history).
- Causal price-elasticity modeling (price is used as a feature, not modeled as a causal lever).
