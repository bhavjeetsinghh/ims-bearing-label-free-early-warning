# Label-free evaluation of bearing early-warning rules (NASA IMS run-to-failure data)

Code and data for:

> Hora, B. S., & Rezapour, S. *Label-Free Evaluation of Bearing Early-Warning Rules on the NASA IMS Run-to-Failure Data: Detection, False Alarms and Maintenance Cost.* Manuscript submitted to the International Journal of Prognostics and Health Management.

## What is here

| Path | Contents |
|---|---|
| `data/bearing3_features.csv` | Test 3 per-snapshot features (Bearing 3 features; RMS of Bearings 1–4), used for the development analysis. |
| `data/test1_features.csv`, `data/test2_features.csv`, `data/test3_features.csv` | Per-snapshot, per-channel features for all three IMS runs, produced by `src/extract_features.py` from the raw NASA files. |
| `src/extract_features.py` | Computes the feature tables from the raw IMS snapshot files. |
| `src/analysis.py` | Test 3 development run: data audit, label-free protocol, detectors, threshold sweep, threshold-label demonstration, renewal-reward cost model. |
| `src/sensitivity.py` | Robustness to alarm persistence and ambiguous-window length. |
| `src/threshold_label_demo.py` | Reproduces the supervised classifiers trained on threshold-derived labels. |
| `src/multi_run.py` | Applies the protocol, unchanged, to all four documented failures (Tests 1–3). |
| `src/decision.py` | Combines detection and false-alarm rates with the cost model. |
| `src/uncertainty_scenarios.py` | Revision analyses: warnings on non-failed bearings counted per bearing, run-level detection interval, Poisson interval for the false-alarm rate, and cost scenarios for warnings on non-failed bearings. |
| `src/tables.py` | Prints Tables 1–4. |
| `src/figures.py` | Figures 1–6 (PNG at 600 dpi and PDF). |

Time is measured in operating hours: the IMS documentation states that longer gaps between files mark resumption of the experiment on the next working day, so each pause longer than 15 min counts as one 10-min interval.

## Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh
```

Tested with Python 3.12, numpy 2.5, pandas 3.0, scipy 1.18, scikit-learn 1.9, matplotlib 3.10.

## Rebuilding the feature tables from raw data (optional)

Download the IMS bearing data from the NASA prognostics data repository, extract the three runs, then:

```bash
python src/extract_features.py 1st_test data/test1_features.csv
python src/extract_features.py 2nd_test data/test2_features.csv
python src/extract_features.py 4th_test/txt data/test3_features.csv
```

## Source data

The raw vibration files are from the IMS Bearing Data Set (Lee, J., Qiu, H., Yu, G., Lin, J., & Rexnord Technical Services, 2007; IMS, University of Cincinnati), distributed through the NASA Ames Prognostics Data Repository. They are not redistributed here; the feature table was computed from them.

## License

Code: MIT (see `LICENSE`). Please cite the paper if you use this work.
