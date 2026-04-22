.DEFAULT_GOAL := help

PYTHON ?= python3.12
VENV   ?= .venv
BIN    := $(VENV)/bin

.PHONY: help venv install install-dev test test-cov lint format typecheck run clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## Create the Python 3.12 virtualenv
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: venv ## Install runtime dependencies
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e .

install-dev: venv ## Install runtime + dev dependencies
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install -e .

test: ## Run the test suite
	$(BIN)/pytest

test-cov: ## Run tests with coverage
	$(BIN)/pytest --cov --cov-report=term-missing

lint: ## Run ruff lint checks
	$(BIN)/ruff check src tests

format: ## Apply ruff + black formatters
	$(BIN)/ruff check --fix src tests
	$(BIN)/black src tests

typecheck: ## Run mypy (strict) on src/
	$(BIN)/mypy src

run: ## Launch the Streamlit app
	$(BIN)/streamlit run src/app/streamlit_app.py

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
