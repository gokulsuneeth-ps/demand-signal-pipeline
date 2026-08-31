.PHONY: setup lint format test cov precommit clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev]" && pre-commit install

lint:
	ruff check src tests

format:
	ruff format src tests

test:
	pytest

cov:
	pytest --cov=src/dsp --cov-report=html
	@echo "Open htmlcov/index.html"

precommit:
	pre-commit run --all-files

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage