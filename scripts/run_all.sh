#!/usr/bin/env bash
# Reproduce every number and figure in the repo (about 25 min on one CPU core).
set -e
cd "$(dirname "$0")/.."
python -m unittest discover -s tests -t . -v
python scripts/run_detection.py --seeds 3
python scripts/run_explanations.py --seeds 3 --alerts 32
python scripts/run_drift.py --seeds 3
python scripts/make_figures.py
