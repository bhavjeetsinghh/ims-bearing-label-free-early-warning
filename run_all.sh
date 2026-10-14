#!/usr/bin/env bash
# Reproduces every number, table and figure in the paper from the feature tables in data/.
set -e
mkdir -p results figures
python src/analysis.py data/bearing3_features.csv results                 # Test 3 development run
python src/sensitivity.py results/series.csv results/results.json results/sensitivity.json
python src/threshold_label_demo.py data/bearing3_features.csv             # supervised threshold-label design
python src/multi_run.py data/test1_features.csv data/test2_features.csv data/test3_features.csv results   # all four failures
python src/decision.py results/results.json results/multi.json results/decision.json
python src/tables.py results
python src/figures.py results results
mv results/Figure*.png results/Figure*.pdf figures/
echo "Done. Results in results/, figures in figures/."
