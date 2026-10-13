.DEFAULT_GOAL := help
PY      := .venv/bin/python
BIN     := .venv/bin
NB      := notebooks/01_data_and_eda.ipynb notebooks/02_features_and_baselines.ipynb \
           notebooks/03_deep_model.ipynb notebooks/04_tournament_simulation.ipynb

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

setup:  ## create .venv (Python 3.12) and install pinned requirements + the package
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -r requirements.txt
	uv pip install --python $(PY) --no-deps -e .

data:  ## download (checksum-verified) and parse the open CC0 match data
	$(PY) -m ucl.data

test:  ## run the pytest suite
	$(BIN)/pytest

notebooks:  ## execute all notebooks top to bottom, in order, keeping outputs
	$(BIN)/jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=3600 $(NB)

lint:  ## ruff lint + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## auto-format
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

freeze:  ## refresh the pinned requirements from the working environment
	uv pip freeze --python $(PY) --exclude-editable > requirements.txt

.PHONY: help setup data test notebooks lint format freeze
