"""FastAPI app serving the registered forecasting model.

Stub for day 1; implemented day 6. Will load the current production model
from the MLflow registry at startup rather than a pickle baked into the
Docker image, so a new model version is a registry promotion, not a rebuild.
"""
