# Analysis scripts

Scripts behind the calibration, significance and graph analyses in the paper. Run them from
anywhere; paths resolve relative to the repo root (override with `CAHGTN_ROOT`; the raw data
folder with `METRO_DATA_DIR`). Outputs are JSON files in `results/` (already committed).

**Inputs.** `Metro_Data/` (raw data), `weights/<City>/*.pt` (script 02 only) and, from the Zenodo
archive, the files in `cache/`:
`Hangzhou_predictions.pkl`, `Hangzhou_windows.npz`, `Shanghai_windows.npz`,
`Shanghai_reduced.npz`, `Beijing_reduced.npz` (float32 per-seed predictions extracted from the
full prediction pickles, which are too large to distribute). Requires `numpy`, `scipy`; scripts
02 and 07 also `torch`.

| Script | What it computes | Paper location | Output |
|---|---|---|---|
| `01_calibration_test.py` | calibration scale `c` fitted on odd test windows / reported on even; 5-block cross-fit | III-G, IV-B, Table III, Fig. 2 | `calibration_test.json` |
| `02_calibration_valfit.py <City>` | `c` fitted on official-validation predictions (Hangzhou, Shanghai), applied to test | IV-B, Table III, abstract | `calibration_valfit_<City>.json` (+ `cache/valpreds_<City>.npz`) |
| `03_calibration_valfit_strict.py` | same, dropping the last 3 validation windows whose targets reach into the test period (run 02 first) | robustness | `calibration_valfit_strict.json` |
| `04_significance_robustness.py` | moving-block bootstrap (blocks 6/12/36/73), Diebold-Mariano with Bartlett bandwidth 6/12/37/74, point-head and single-graph comparisons | III-G, IV-B/C, ablations | `significance_robustness.json` |
| `05_graph_edge_overlap.py` | edge overlap between the graph fitted on all training days and on internal-fit days only (88% / 93% / 68%) | III-B | `graph_edge_overlap.json` |
| `06_ensemble_check.py` | coverage of the moment-matched ensemble vs the exact seed mixture, horizons 1-4 | III-G | `ensemble_check.json` |
| `07_graph_sensitivity.py --city <City>` | phase-1 retraining under the full-train graph vs the internal-fit-only graph (5 paired seeds; hours on CPU) | III-B, Discussion | `graph_sensitivity_<City>.json` |

`results/graph_sensitivity_summary.json` summarises the three cities (internal-fit minus full-graph
15-min validation MAE: Hangzhou +0.20, Shanghai -0.01, Beijing +0.04; paired SD 0.24-0.33).
`results/ensemble_check_horizons_*_original.json` are the earlier two-run versions of script 06.

The scripts print progress and write only inside `analysis/results/` and `cache/`.
