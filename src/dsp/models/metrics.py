"""WAPE / MAPE / RMSE — defined once, imported everywhere they're reported.

Stub for day 1; implemented day 4. Having one definition of each metric
(rather than reimplementing WAPE inline wherever it's needed) is what keeps
the backtest harness, the MLflow logging, and the dashboard from silently
disagreeing with each other about what "16% error" means.
"""
