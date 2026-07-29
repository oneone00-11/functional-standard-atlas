#!/bin/bash
# Full-matrix SpliceAI scoring driver — resumable, ~1-4 h on a 10-core laptop.
# Reruns skip completed chunks (cache/chunks/*.done, md5-keyed).
# Usage: models/spliceai/run_full.sh [extra score.py args]
set -e
cd "$(dirname "$0")/../.."
export TF_CPP_MIN_LOG_LEVEL=3
PYTHONPATH=src models/spliceai/.venv/bin/python models/spliceai/score.py \
  --input data/frozen/frozen-matrix-v1.parquet \
  --output results/spliceai_ds_scores.parquet "$@"
