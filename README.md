# Label-free evaluation of bearing early-warning rules (NASA IMS Test 3, Bearing 3)

Code and data for:

> Hora, B. S., & Rezapour Behnagh, S. *Label-Free Evaluation of Bearing Early-Warning Rules and Their Maintenance-Cost Implications: A NASA IMS Run-to-Failure Case Study.* Manuscript submitted to the International Journal of Prognostics and Health Management.

## What is here

| Path | Contents |
|---|---|
| `data/bearing3_features.csv` | Per-snapshot features for all 6,324 files of IMS Test 3 (RMS, Pearson kurtosis, skewness, variance, peak-to-peak for Bearing 3; RMS for Bearings 1–4). The last row (2004.04.18.02.42.55) is an invalid snapshot and is excluded by the analysis. |
| `src/analysis.py` | Data audit, label-free evaluation protocol, detectors D1–D4, threshold sweep, cross-bearing check, threshold-label (leakage) demonstration, renewal-reward cost model. Writes `results/results.json` and `results/series.csv`. |
| `src/sensitivity.py` | Robustness to alarm persistence (k = 1, 3, 5) and ambiguous-window length (3, 7, 10 days). |
| `src/threshold_label_demo.py` | Reproduces the supervised classifiers trained on threshold-derived labels. |
| `src/figures.py` | Figures 1–6 (PNG at 600 dpi and PDF). |
| `results/` | Outputs used in the paper. |
| `figures/` | Figures as they appear in the paper. |

## Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh
```

Tested with Python 3.12, numpy 2.5, pandas 3.0, scipy 1.18, scikit-learn 1.9, matplotlib 3.10.

## Source data

The raw vibration files are from the IMS Bearing Data Set (Lee, J., Qiu, H., Yu, G., Lin, J., & Rexnord Technical Services, 2007; IMS, University of Cincinnati), distributed through the NASA Ames Prognostics Data Repository. They are not redistributed here; the feature table was computed from them.

## License

Code: MIT (see `LICENSE`). Please cite the paper if you use this work.
