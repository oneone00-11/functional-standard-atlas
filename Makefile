PYTHON ?= PYTHONPATH=src python3

.PHONY: setup fetch test clean

setup:
	pip install -r requirements.txt

fetch:
	$(PYTHON) -m atlas.fetch_mavedb --config config/assays.yaml

test:
	$(PYTHON) -m pytest tests/ -q

clean:
	rm -rf results/*
