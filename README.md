# CA-HGTN: Calibrated Distributional Metro Ridership Forecasting

Code, trained weights, and analysis scripts for **"Calibrated Distributional Metro
Ridership Forecasting with Multi-Graph Neural Networks"** (AAIML 2027).

CA-HGTN forecasts station-level metro inflow/outflow 15–60 minutes ahead and predicts a
full **Negative-Binomial distribution** per station and horizon instead of a single
number. It fuses three graph views (physical, lagged-correlation, adaptive) through a
learned gate, and encodes time with a Transformer + BiGRU.

> **Paper:** [link to PDF / arXiv, once available]  
> **Archived release (code, weights, predictions):** [DOI]

## What is in this repository

```
cahgtn/                  model, data pipeline, training, evaluation, significance tests
notebooks/               [data pipeline, training/search, evaluation & ablations]
weights/                 trained CA-HGTN and GCN+GRU checkpoints (5 seeds, Hangzhou & Shanghai)
analysis/                scripts behind the calibration and significance analyses in the paper
  partA.py               test-set calibration (odd/even split, 5-block cross-fit)
  partB.py               calibration fitted on official-validation predictions
  partC.py               block-bootstrap and long-bandwidth Diebold-Mariano checks
  partD_overlap.py       correlation-graph edge overlap (all-train vs internal-fit)
  partE.py               moment-matched ensemble vs exact mixture coverage
  graph_sensitivity.py   phase-1 retraining under the two graphs
  results/               JSON outputs of the above
```

Large files (prediction pickles, windowed arrays) are in the Zenodo archive, not here.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy pandas scipy        # [pin the versions you used, e.g. torch==2.x]
```
Tested with Python [3.x] and PyTorch [x.x] on [macOS / Apple Silicon (MPS)].
The CPU path works but is slow (Shanghai has 288 stations).

## Data

Raw data is not redistributed. All three datasets use the LibCity atomic-file format
(`.dyna`, `.rel`).

| City | Description | Source |
|---|---|---|
| Hangzhou (HZMetro) | 80 stations, 15-min, Jan 2019; official split from PVCGN | Liu et al., IEEE T-ITS 2020 [link to dataset page] |
| Shanghai (SHMetro) | 288 stations, 15-min, Jul–Sep 2016; official split from PVCGN | Liu et al., IEEE T-ITS 2020 [link to dataset page] |
| Beijing Subway | 276 stations, 15-min; no published split (chronological 70/15/15 used here) | Zhang et al., IEEE T-ITS 2020, via LibCity [link] |

Place the files as `Metro_Data/HZMETRO/`, `Metro_Data/SHMETRO/`,
`Metro_Data/BEIJING_SUBWAY_15MIN/`. Please follow each dataset's own terms of use.

Splits: Hangzhou and Shanghai use the official train/validation/test splits. Beijing uses
a 70/15/15 chronological split, and its final refit uses the leading 85%, so it has no
held-out validation segment (see the paper's Dataset section).

## Reproducing the results

1. **Data pipeline:** `notebooks/[00_Data_Pipeline.ipynb]` builds windows and graphs.
2. **Training (two phases):** phase 1 selects the epoch budget on internal-validation;
   phase 2 refits on the full official training period for that many epochs and is
   evaluated on test once.
3. **Evaluation and ablations:** `notebooks/[02_Evaluate_and_Ablate.ipynb]`.
4. **Calibration and significance analyses:**
   ```bash
   export SUBWAY_ROOT=/path/to/project   # folder containing cahgtn/, Metro_Data/, weights
   cd analysis && python partA.py && python partC.py
   ```
   Pretrained weights in `weights/` let you skip training. File names follow
   `{City}_CA-HGTN_'full'_seed{k}.pt`.

## Key results (15-min horizon, Hangzhou, 5 seeds, standard protocol)

[Copy 2–3 lines from Table II/IV of the paper, e.g., CA-HGTN vs GCN+GRU MAE and
uncorrected 90% coverage per city.]

## Important caveats (also stated in the paper)

- Point accuracy trails the best published forecasters on this benchmark; the
  contribution is calibrated distributional output and a rigorous evaluation protocol.
- Uncorrected 90% intervals are overconfident (coverage 0.786–0.886). The post-hoc scalar
  correction is an **internally-validated diagnostic, not a held-out guarantee**:
  fitted on separate validation days it reaches nominal coverage on Hangzhou but not
  Shanghai.
- The correlation graph is built once from the official training period, so it is not
  blind to the internal-validation days; test data never enters the graph, weights, or
  epoch budget.
- Shanghai WMAPE and MAPE are not interchangeable; see the paper before comparing to
  external percentage figures.

## Citation

```bibtex
@inproceedings{yoon2027cahgtn,
  title     = {Calibrated Distributional Metro Ridership Forecasting with Multi-Graph Neural Networks},
  author    = {Yoon, Liam},
  booktitle = {Proceedings of AAIML 2027},
  year      = {2027}
}
```

## License

[MIT / Apache-2.0]. The datasets are subject to their own licenses.

## Contact

Liam Yoon, Seoul, South Korea. liam.yoon.6429@gmail.com
