# CA-HGTN: Calibrated Distributional Metro Ridership Forecasting

Code, trained weights, and analysis scripts for **"Calibrated Distributional Metro Ridership
Forecasting with Multi-Graph Neural Networks"** (Liam Yoon, AAIML 2027).

CA-HGTN forecasts station-level metro inflow/outflow 15-60 minutes ahead and predicts a full
**Negative-Binomial distribution** per station and horizon instead of a single number. It fuses
three graph views (physical, lagged-correlation, adaptive) through a learned gate and encodes time
with a Transformer + BiGRU.

> **Paper:** [link to PDF, once available]
> **Archived release (code, weights, large prediction files):** DOI: 10.5281/zenodo.24103658

## Repository layout

```
00_Data_Pipeline.ipynb       build windows, seasonal profiles and graphs per city
01_Train_CAHGTN.ipynb        two-phase training, baselines, ablations
02_Evaluate_and_Ablate.ipynb metrics, significance tests, calibration
cahgtn/                      model, data pipeline, training, evaluation, significance tests
weights/<City>/*.pt          trained checkpoints (5 seeds x variants x 3 cities)
analysis/                    scripts behind the paper's calibration / significance analyses
  results/                   their JSON outputs (committed)
docs/METHODOLOGY.md          the method and protocol in detail
figures/                     example station-level forecasts
```

The notebooks write their outputs (`{City}_windows.npz`, `{City}_predictions.pkl`, `*.pt`) to the
repo root, as in the original project. Large files (predictions, windows) are in the Zenodo archive,
not in this repository.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Developed on Python 3.13.9 / PyTorch 2.12.1 / macOS with Apple-silicon MPS. CPU works but is
slow, especially for Shanghai (288 stations).

## Data

Raw data is **not redistributed**. Obtain it from the original sources and place it so that
`Metro_Data/` contains `HZMETRO/`, `SHMETRO/` and `BEIJING_SUBWAY_15MIN/` (LibCity atomic files:
`.dyna`, `.rel`, `.geo`, `config.json`). To keep it elsewhere, set `METRO_DATA_DIR` in notebook 00
or the environment variable `METRO_DATA_DIR` for the analysis scripts.

| City | Description | Source |
|---|---|---|
| Hangzhou (HZMetro) | 80 stations, 15-min, Jan 1-25 2019; official split from PVCGN | [PVCGN repository](https://github.com/ivechan/PVCGN) (Liu et al., IEEE T-ITS 2020) |
| Shanghai (SHMetro) | 288 stations, 15-min, Jul 1-Sep 30 2016; official split from PVCGN | [PVCGN repository](https://github.com/ivechan/PVCGN) |
| Beijing Subway | 276 stations, 15-min; no published split (a chronological 70/15/15 split is used) | [ResNet-LSTM-GCN repository](https://github.com/JinleiZhangBJTU/ResNet-LSTM-GCN) (Zhang et al., IEEE T-ITS 2020) |

All three are used in [LibCity](https://github.com/LibCity/Bigscity-LibCity) format
([dataset documentation](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/data/raw_data.html)).
The data sources state no license on those pages; check each project's terms before redistributing.
Please cite the original dataset papers and LibCity (see Citation).

Splits (details in `docs/METHODOLOGY.md`): Hangzhou and Shanghai use the official splits. Beijing's
final refit and correlation graph use the leading 85% of the series, so it has no held-out
validation segment.

## Reproducing

1. **Windows and graphs:** run `00_Data_Pipeline.ipynb` for each city (set `CITY`).
2. **Training:** run `01_Train_CAHGTN.ipynb` for each city. Phase 1 chooses the epoch budget on an
   internal validation split; phase 2 refits on the full official training period and is evaluated
   on test once. Results go to `{City}_predictions.pkl`.
3. **Evaluation:** run `02_Evaluate_and_Ablate.ipynb`.
4. **Analysis scripts** (calibration protocols, block-aware significance tests, graph checks) are
   documented in (analysis/README.md).

The trained checkpoints in `weights/` let you skip training for the analyses that need models.

## Key results

See the paper for all numbers and caveats. In brief: on Hangzhou, CA-HGTN's 15-minute MAE is 26.74
(standard protocol) and 24.23 (extended training) versus 28.54 / 28.17 for a reproduced GCN+GRU
baseline; the improvement holds on Hangzhou and Shanghai under block-aware tests but is not
established on Beijing.

## Caveats (also stated in the paper)

- Point accuracy trails the best published number for this benchmark (PB-GRU, 22.13 on Hangzhou);
  the contribution is calibrated distributional output and a more rigorous evaluation protocol.
- Uncorrected 90% intervals are overconfident (coverage 0.786-0.886 across cities). The post-hoc
  dispersion scalar is an **internally-validated diagnostic, not a held-out guarantee**: fitted on
  separate validation days it reaches nominal coverage on Hangzhou but not Shanghai.
- The correlation graph is fitted once on the full official training period, so it is not blind to
  the internal-validation days; no test-period data enters the graph, weights or epoch budget.
- Shanghai WMAPE and MAPE are not interchangeable; see the paper before comparing to external
  percentage figures.

## Citation

```bibtex
@inproceedings{yoon2027cahgtn,
  title     = {Calibrated Distributional Metro Ridership Forecasting with Multi-Graph Neural Networks},
  author    = {Yoon, Liam},
  booktitle = {Proceedings of AAIML 2027},
  year      = {2027}
}
```
Please also cite LibCity and the dataset papers (PVCGN; ResNet-LSTM-GCN):

```bibtex
@inproceedings{libcity2021,
  author    = {Wang, Jingyuan and Jiang, Jiawei and Jiang, Wenjun and Li, Chao and Zhao, Wayne Xin},
  title     = {LibCity: An Open Library for Traffic Prediction},
  booktitle = {Proceedings of the 29th International Conference on Advances in Geographic Information Systems (SIGSPATIAL)},
  year      = {2021},
  pages     = {145--148},
  doi       = {10.1145/3474717.3483923}
}
```

## License

MIT (see `LICENSE`) for the code. The datasets are subject to their own terms.
