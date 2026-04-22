.DEFAULT_GOAL := help

PYTHON ?= python3.12
VENV   ?= .venv
BIN    := $(VENV)/bin

.PHONY: help venv install install-dev test test-cov lint format typecheck run samples clean

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
	$(BIN)/ruff check src tests scripts

format: ## Apply ruff + black formatters
	$(BIN)/ruff check --fix src tests scripts
	$(BIN)/black src tests scripts

typecheck: ## Run mypy (strict) on src/ + tests/
	$(BIN)/mypy src tests scripts

run: ## Launch the Streamlit app
	$(BIN)/streamlit run src/app/streamlit_app.py

samples: ## Regenerate the sample datasets under data/samples/
	$(BIN)/python scripts/generate_sample_data.py

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
