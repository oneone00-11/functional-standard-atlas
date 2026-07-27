# Prefer the project virtualenv when it exists; fall back to system python3.
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
export PYTHONPATH := src

.PHONY: setup fetch freeze test clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

fetch:
	$(PY) -m atlas.fetch_mavedb --config config/assays.yaml

freeze:
	$(PY) -m atlas.mapping

test:
	$(PY) -m pytest tests/ -q

clean:
	rm -rf results/*
