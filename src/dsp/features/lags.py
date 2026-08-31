"""Lag and rolling-window features.

Stub for day 1; implemented day 3. Watch for leakage here specifically: any
rolling stat must only look backward from the forecast origin, never
forward — this is the single most common bug in demand-forecasting feature
pipelines and the first thing a reviewer should check for.
"""
