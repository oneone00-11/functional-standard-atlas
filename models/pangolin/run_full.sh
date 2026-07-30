#!/bin/bash
# Full-matrix Pangolin scoring driver — resumable, multi-hour torch CPU job.
# Reruns skip completed chunks (cache/chunks/*.done, md5-keyed).
# Usage: models/pangolin/run_full.sh [extra score.py args]
set -e
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHONPATH=src models/pangolin/.venv/bin/python models/pangolin/score.py \
  --input data/frozen/frozen-matrix-v1.parquet \
  --output results/pangolin_scores.parquet "$@"
