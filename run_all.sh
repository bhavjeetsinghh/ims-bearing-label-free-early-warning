#!/usr/bin/env bash
# Reproduces every number, table and figure in the paper.
set -e
mkdir -p results
python src/analysis.py data/bearing3_features.csv results
python src/sensitivity.py results/series.csv results/results.json results/sensitivity.json
python src/threshold_label_demo.py data/bearing3_features.csv
python src/figures.py results
mv results/Figure*.png results/Figure*.pdf figures/ 2>/dev/null || true
echo "Done. Results in results/, figures in figures/."
