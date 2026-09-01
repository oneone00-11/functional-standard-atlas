# Prefer the project virtualenv when it exists; fall back to system python3.
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
export PYTHONPATH := src

# The pinned scientific stack needs Python >= 3.12 (numpy 2.5.1 and scipy
# 1.18.0 both declare it). A bare `python3` is frequently older -- macOS still
# ships 3.9 -- and pip then fails with "no matching distribution found for
# pandas==3.0.5", which points at the wrong thing entirely. Check first and say
# what is actually wrong. Override with:  make setup PYTHON=python3.12
PYTHON ?= python3
MIN_PY := 3.12

.PHONY: setup fetch freeze test clean check-python artefacts check-numbers facts verify

check-python:
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 12) else 1)' \
	  || { \
	    echo ""; \
	    echo "ERROR: '$(PYTHON)' is $$($(PYTHON) -c 'import sys;print("%d.%d"%sys.version_info[:2])'), but the pinned"; \
	    echo "       dependencies require Python >= $(MIN_PY) (numpy 2.5.1, scipy 1.18.0)."; \
	    echo ""; \
	    echo "       Re-run with a newer interpreter, for example:"; \
	    echo "           make setup PYTHON=python3.12"; \
	    echo ""; \
	    exit 1; \
	  }

setup: check-python
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

fetch:
	$(PY) -m atlas.fetch_mavedb --config config/assays.yaml

freeze:
	$(PY) -m atlas.mapping

test:
	$(PY) -m pytest tests/ -q

# Rehash every delivered figure and table into manifests/artefacts.json. Run
# after a rebuild; `make test` then fails if anything drifts from the record.
artefacts:
	$(PY) -m atlas.artefact_manifest --write

# Does every number in the manuscript come from pipeline output? Tokens with no
# counterpart are reported and fail the run unless they are listed, with a
# reason, in config/manuscript_number_whitelist.yaml.
#     make check-numbers DOCX=path/to/manuscript.docx
DOCX ?=
check-numbers:
	@test -n "$(DOCX)" || { echo "usage: make check-numbers DOCX=path/to/manuscript.docx"; exit 2; }
	$(PY) -m atlas.check_manuscript_numbers "$(DOCX)" --verbose

# Counts the manuscript cites that live in no results/ file -- the guardrail
# suite size above all. Without this a stale "102 guardrail tests" would match
# some unrelated pipeline value by coincidence and pass as verified.
facts:
	$(PY) -m atlas.pipeline_facts --write

# Everything a delivered artefact has to satisfy.
verify: test
	$(PY) -m atlas.artefact_manifest --check
	$(PY) -m atlas.pipeline_facts --check

clean:
	rm -rf results/*
