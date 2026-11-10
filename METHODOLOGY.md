# Methodology

This document summarizes the methodology of the paper ("Calibrated Distributional
Metro Ridership Forecasting with Multi-Graph Neural Networks", AAIML 2027); the paper is the
authoritative source for every number.

## 1. Problem formulation

Given a lookback window of `T=12` past 15-minute intervals (3 hours) of station-level
inflow/outflow counts across `N` stations, predict a full probability distribution over
inflow/outflow at each of the next `H=4` horizons (15/30/45/60 minutes ahead), not just a
point estimate. Ridership counts are non-negative, over-dispersed (variance exceeds the
mean, especially at high-traffic stations and during rush hours), and strongly periodic
(time-of-day, day-of-week). The method is designed around these three properties.

## 2. Data and splits

LibCity-format datasets (HZMetro, SHMetro, Beijing Subway), 15-minute aggregation. Splits
are date-anchored (or chronological), never randomly shuffled, to avoid leakage between
temporally adjacent windows.

| | Stations | Train | Official val | Test | Internal fit | Internal val |
|---|---|---|---|---|---|---|
| HZMetro | 80 | 2019-01-01 to 01-18 | 01-19 to 01-20 (weekend only) | 01-21 to 01-25 (weekdays) | 01-01 to 01-11 | 01-12 to 01-18 |
| SHMetro | 288 | 2016-07-01 to 08-31 | 09-01 to 09-09 | 09-10 to 09-30 | 07-01 to 08-21 | 08-22 to 08-31 |
| Beijing | 276 | leading 70% (chronological) | next 15% | final 15% | first 80% of the leading 70% | last 20% of the leading 70% |

Test sets contain 362 (Hangzhou), 1,530 (Shanghai), and 267 (Beijing) windows.

**Beijing has no published date split**, so a chronological 70/15/15 split is used. This is
a fallback, not an established convention. For Beijing the final refit and the correlation
graph use the leading **85%** (training plus its validation block), so Beijing has **no
held-out validation segment**; analyses that need one (e.g. fitting the calibration scale on
validation predictions) are unavailable for Beijing.

"Internal fit/val" is a strict subset of "train", used only for epoch-budget (and
learning-rate) selection (§5). Internal validation periods mix weekday and weekend days so
that selection is not tuned against an unrepresentative validation distribution. The official
validation period is not used for any tuning; for Hangzhou and Shanghai it is used only for
one diagnostic (§6, validation-fit calibration scale).

Three graph structures are available as spatial inputs:
- **Physical**: station adjacency from the transit network topology.
- **Correlation**: a directed, multi-lag cross-correlation graph (lags 0 to 8 intervals,
  top-8 incoming edges per station), built from deseasonalized flow series (subtracting each
  station's own historical per-timeslot mean before computing correlation, so shared
  rush-hour timing does not dominate every edge at lag 0).
  **The graph is fitted once, on the full official training period, which includes the
  internal-validation days.** It is therefore *not blind* to those days, and the two-phase
  protocol's leakage-safety guarantee (§5) covers the model weights and epoch budget, not this
  fixed graph. No test-period data enters the graph. Sensitivity checks: a graph fitted on
  internal-fit days only keeps 88%, 93%, and 68% of the edges (Hangzhou, Shanghai, Beijing),
  and retraining phase one under it changes 15-minute internal-validation MAE by +0.20, -0.01,
  and +0.04 (internal-fit minus full-graph; paired SD 0.24-0.33 over five seeds), none
  distinguishable from zero. This checks phase one only and does not make the graph blind to
  those days.
- **Precomputed similarity/correlation**: the dataset's own `.rel`-file columns, reported in the
  paper as an exact match to PVCGN's original graph construction (a claim taken from the project
  notes; not independently re-verified here). Available as an alternative spatial input; see §8.

## 3. Architecture (CA-HGTN)

1. **Input encoding**: raw counts are log1p-compressed, linearly projected to `hidden_dim`
   (default 64), and added to a learned time-of-day/day-of-week embedding. Per-station
   context features (seasonal mean and daily-lag flow at "now", log1p-compressed and
   projected) are added after temporal pooling, followed by a small trunk.
2. **Spatial layers** (2, stacked): multi-graph attention over stations at each lookback
   timestep. Up to three graphs are fused via a learned, softmax-normalized gate: physical
   (row-normalized, cached since it is fixed), correlation (same), and an adaptive graph
   (`softmax(relu(E1 @ E2^T))` over learned per-station embeddings, recomputed every call
   since it depends on trainable parameters). The fused graph enters scaled-dot-product
   attention as an additive log-bias, not a multiplicative mask; attention's own softmax
   renormalizes regardless of the bias's row-sum. `graph_mode` selects the subset of graphs
   (`full`, `physical_only`, `correlation_only`, `adaptive_only`, `no_physical`), supporting
   the ablations in §7. (`no_physical` sums the correlation and adaptive graphs without a
   gate.)
   - *Optional, off by default*: `edge_dropout` randomly zeros a fraction of each graph's
     edges every training step (train mode only), adapting DropEdge (Rong et al., ICLR 2020);
     and an adaptive-graph entropy regularizer.
3. **Temporal fusion**: a Transformer encoder (2 layers, 4 heads) and a bidirectional GRU
   process the lookback sequence in parallel; their outputs are combined via a learned gate,
   then attention-pooled across the lookback window.
4. **Output head, seasonal-residual decomposition**: rather than predicting ridership from
   scratch, the model predicts a *bounded multiplicative correction* around each station's
   own historical mean for that horizon's time-of-day slot:
   ```
   mu = (seasonal_baseline + 1.0) * exp(clamp(correction, -3, 3))
   ```
   The correction head is zero-initialized, so training starts exactly at the seasonal
   baseline. The `+1.0` floor avoids multiplying by exactly zero for genuinely zero-traffic
   (station, timeslot) cells; the `[-3, 3]` clamp bounds the correction to roughly
   `[0.05x, 20x]` of the seasonal baseline. The seasonal profile is fitted on the fitting
   data of each phase (internal-fit in phase 1, the full official training period in phase 2).
5. **Distributional head**: `mu` is the Negative Binomial mean via the mechanism above. The
   dispersion `r` comes from a separate linear head,
   `r = softplus(linear(h)) + 1e-3`, with its bias initialized so that `r` starts near 10
   (moderate dispersion). Variance is `mu + mu^2/r`: larger `r` means lower over-dispersion,
   approaching Poisson as `r -> infinity`. (`r` is predicted directly; it is *not* a
   correction around a reference value.)

## 4. Loss function

**Negative Binomial negative log-likelihood** is the primary training signal for `mu` and `r`
jointly, but is not used alone. Plain NB-NLL has a failure mode: the model can lower its loss
by shrinking `r` (declaring high uncertainty) for a hard-to-predict cell instead of improving
`mu`. Verified numerically: for a badly wrong `mu=254` against a true value of 3809, shrinking
`r` from 10 to 0.5 cuts the loss from 118.4 to 15.3 (7.7x) without correcting the mean. Two
combinable mechanisms address this:

- **Auxiliary point loss** (`aux_weight=0.15` by default): a target-magnitude-weighted Huber
  loss on `mu` directly (weight = target^0.5), added to the NB-NLL. Its gradient does not
  depend on `r`, so it keeps pressure on `mu` regardless of what `r` does.
- **Beta-NLL reweighting** (`beta`, off by default at 0): adapts Seitzer et al. (ICLR 2022)
  from their Gaussian heteroscedastic case to the Negative Binomial one, multiplying each
  cell's NLL by `(r.detach() / r_reference) ** (-beta)` (`r_reference=10.0`, matching the
  dispersion-head initialization), which up-weights cells where `r` has shrunk. `r.detach()`
  prevents the reweighting from creating a new feedback loop into `r`. In the scenario above
  it neutralizes the escape hatch at about `beta=0.75` (loss ratio 0.82 at 0.75, 0.39 at 1.0)
  and reverses it by roughly 50x at `beta=2`. See §8 for how it performs in practice when it
  *replaces* the auxiliary term.

## 5. Training protocol

Two-phase, to avoid tuning the epoch budget on data used in the final model:

1. **Phase 1 (epoch-budget selection)**: train on `internal_fit`, early-stop against
   `internal_val`. Early stopping requires `min_epochs_before_stopping=25` epochs before it can
   trigger, and a relative improvement of at least `min_delta_rel=0.15%` to reset patience
   (`patience=15` epochs without such an improvement), deliberately conservative because the
   datasets are small (roughly 1,000 to 4,500 overlapping training windows depending on the
   city) and naive early stopping is prone to noise-driven premature stops. The learning rate
   is chosen from `{1e-3, 3e-4}` by internal-validation loss (notebook `01_Train_CAHGTN`).
2. **Phase 2 (final refit)**: retrain from scratch on the full official `train` period (the
   leading 85% for Beijing) for exactly the epoch count selected in phase 1, with no further
   validation signal. The official test set is touched once, at the end.

`AdamW`, `weight_decay=1e-3`, linear LR warmup (5 epochs, from 10% to 100% of the base rate)
then cosine decay to zero, gradient clipping at norm 5.0. The standard protocol is repeated
across 5 independent seeds; each seed independently repeats both phases (a different seed can
genuinely need a different epoch budget, and sharing budgets across seeds would compromise the
independence that multi-seed evaluation exists to measure).

**Extended training (validation-searched).** Early configurations chose longer epoch ceilings by
watching test MAE, which is structurally equivalent to early-stopping on the test set. The
replacement is a validation-based search over `(patience, min_epochs_before_stopping, ceiling)`
candidates (`search_patience_and_ceiling`; the training notebook currently lists
`(100, 175, 350)`, `(250, 300, 500)`, `(350, 350, 600)`), scored on internal validation, followed by
the same phase-2 refit. Extended-training results use 5 refit seeds on Hangzhou and 3 on
Shanghai and Beijing. Extended training improves CA-HGTN's Hangzhou 15-minute MAE by 9.4%
(26.74 to 24.23) but worsens uncorrected calibration (§8).

## 5b. A deliberate exception to the above, clearly separated

`diagnostic_train_with_test_trajectory` (`train.py`) breaks the "never touch the test set
during training" rule on purpose, for one question: does long, unstopped training on this
dataset actually overfit? It trains for a long fixed budget (default 300 epochs, approaching
PB-GRU's published 350) with early stopping disabled, checking test MAE periodically to trace
its trajectory. This is legitimate only as an observational diagnostic: its output must never
be used to select an epoch count, checkpoint, or hyperparameter for anything reported as an
official result. It is kept structurally separate: a distinctly named function, a separate
output file, deliberately excluded from the aggregation notebook.

## 6. Evaluation

- **Point metrics**: MAE, RMSE, MAPE (targets with |y| < 1 excluded), WMAPE. MAPE and WMAPE are
  not interchangeable, which matters when comparing Shanghai to external figures (see the
  paper's SHMetro comparison).
- **Distributional metrics**: coverage at 90% and 50% nominal (do X% intervals contain X% of
  observations), always reported with mean interval width; NB negative log-likelihood; and NASS
  (a skill score adapting Murphy 1996 / Nash-Sutcliffe 1970 / Gneiting & Raftery 2007, using a
  model-endogenous reference floor from fitted dispersion). Uncorrected coverage on the full
  test set is the primary distributional result.
- **Ensembling**: predictions are averaged across seeds (exact for point accuracy). The
  dispersion of the ensemble is moment-matched: the seed-mixture variance is
  `E[Var(seed)] + Var[E(seed)]` (law of total variance), converted back to a single NB `r`. This
  is an approximation to the exact NB mixture; its coverage was checked against the exact
  mixture and agrees within 0.002 across horizons.
- **Post-hoc calibration (secondary diagnostic)**: one scalar `c` rescales dispersion only,
  `r_cal = r / c`, found by binary search to match 90% coverage; `mu` and every point metric are
  untouched. It is fitted three ways: (a) on odd-indexed test windows and reported on the
  even-indexed half (weakest guarantee: both halves share a test period and adjacent windows
  overlap); (b) by cross-fitting over five contiguous test blocks; (c) for Hangzhou and Shanghai,
  on official-validation predictions the models never trained on, applied to the full test set.
  Beijing cannot do (c). All are **internally-validated diagnostics, not held-out guarantees**.
- **Significance testing**: Diebold-Mariano (Bartlett-kernel HAC variance truncated at lag h-1,
  Harvey-Leybourne-Newbold correction; no lags are used at the 15-minute horizon) and Wilcoxon
  signed-rank. The code's self-test documentation targets a ~5% false-positive rate under no
  true difference. **Both tests, and the window bootstrap, treat windows as independent although
  paired window errors are strongly autocorrelated** (lag-1 autocorrelation 0.77 to 0.97), so they
  overstate precision. Block-aware checks are therefore also reported: moving-block bootstrap
  (blocks of 6/12/36/73 windows) and DM with Bartlett bandwidths of 6/12/37/74 windows.
- **Relative improvement**: bootstrap confidence intervals (resampling individual test windows with
  replacement, 5,000 repetitions) for percentage improvement over a specific baseline, not a bare
  point percentage and not a delta-method approximation over only 5 seeds.
- **Stratified-by-deviation analysis**: metrics computed separately on the top 20% of test
  windows by deviation from the seasonal baseline versus the rest, to check whether a model's
  advantage concentrates in anomalous or routine conditions. See §8 for the (mixed) result.

## 7. Baselines

- **GRU-only**: the same seasonal-residual output mechanism and training protocol, no spatial
  structure at all.
- **GCN+GRU**: a Kipf & Welling-style graph convolution over the physical graph plus a GRU, same
  output mechanism and protocol, reproduced under this project's own pipeline to get an
  apples-to-apples comparison point, since cross-paper numbers for the same cited model were
  found to disagree substantially (§8). Optional temporal variants
  (`temporal_mode`, which supersedes the older `bidirectional_attention=True` flag kept as a
  backward-compatible alias):
  - `bidirectional_attention`: a bidirectional GRU with learned attention pooling over the
    lookback window instead of a plain unidirectional GRU with last-hidden-state pooling;
    ~2.0x parameters at the default hidden size.
  - `stacked_bidir_uni_attention` (~3.4x parameters): adds a unidirectional GRU layer after the
    bidirectional one before pooling, a reasonable interpretation (not a verified exact
    replication; source code unavailable) of PB-GRU's description of its temporal component.
  - For comparison, scaling `hidden_dim` from 64 to 256 gives ~15.3x parameters.
- **Architectural ablations** (via `graph_mode` and `head_type`): single-graph (physical topology
  only), no-physical (correlation + adaptive, no physical anchor), and point-head (removes the
  distributional head and auxiliary-loss structure, training on the point loss alone). Further
  variants: beta-NLL isolated (beta-reweighting replacing the auxiliary term) and the
  precomputed-graph substitutions.

Parameter counts (`param_counter.py`): CA-HGTN 211,094 at `hidden_dim=64`, N=80; GCN+GRU
25,992; GRU-only 25,928.

## 8. Current results and open findings

Full numbers are in the paper; summarized here. All MAEs are 15-minute, riders per window.

- **Hangzhou, main result.** CA-HGTN 26.74 +- 0.33 (standard) and 24.23 +- 0.53 (extended) versus
  GCN+GRU 28.54 +- 0.34 and 28.17 +- 0.21, and GRU-only 28.43 +- 0.09 and 28.30 +- 0.35. Relative
  improvement over GCN+GRU under the standard protocol is 5.8% (window-bootstrap 95% CI [4.8,
  6.7]; with 12-window moving blocks [3.5, 8.3]). DM and Wilcoxon p < 0.0001 at 15 minutes;
  DM with a 37-window bandwidth gives p <= 0.0004 at every horizon.
- **Shanghai and Beijing** (narrower protocol). Improvement over GCN+GRU: Shanghai 12.0% [11.1,
  12.8] (block-aware [9.6, 14.6]), robust. Beijing 2.7% [1.8, 3.7] with the window bootstrap, but
  [-0.1, 5.3] with 12-window blocks and DM p = 0.25 to 0.27 at a 37-window bandwidth. The Beijing
  improvement is **consistent in direction but not established**.
- **Calibration.** Uncorrected 90% intervals cover 87.6% (Hangzhou), 78.6% (Shanghai), and 88.6%
  (Beijing): mildly overconfident on two cities and substantially on Shanghai. A scalar fitted on
  one test half reaches nominal coverage, but per-block coverage varies by up to 0.29 on
  Shanghai, and fitted on official-validation days it restores Hangzhou (Cov90 0.893 to 0.896) but
  not Shanghai (0.822 to 0.825). The same scalar over-corrects the 50% level, so one dispersion
  scalar does not calibrate the whole distribution. These are internally-validated diagnostics,
  not evidence the correction generalizes.
- **Precomputed graphs.** Swapping in the dataset's own precomputed (PVCGN-style) graphs for this
  project's correlation construction made Hangzhou results worse (27.60 / 27.59 vs 26.74), despite
  being described as the construction the compared literature uses. Likely explanation: those
  graphs were tuned for a different architecture (PVCGN/PB-GRU use different normalization per
  graph type, while this model applies uniform row-normalization), so matching graph *data* is not
  matching graph *usage*.
- **Beta-NLL in isolation.** Replacing the auxiliary term with beta-reweighting gave 29.64 vs
  26.74 (+2.90) on Hangzhou. The paper reports it degraded accuracy, training stability, and calibration together
  (only the accuracy figure is re-quoted here). The escape-hatch mechanism works as verified
  numerically; the auxiliary term evidently provides broader training-stability value. Combining
  both is untested.
- **Single-graph is a Hangzhou finding.** Single-graph (physical only) is 4.69 MAE worse than the
  full model on Hangzhou, but on Shanghai it is 1.8% *lower* and on Beijing 3.5% higher, neither
  distinguishable from zero with a 37-window DM bandwidth (p = 0.26, 0.19).
- **Point head.** Dropping the NB head costs accuracy on Hangzhou (+1.09), but on Shanghai and
  Beijing the point-head variant has *lower* raw MAE than the full model (Shanghai by 2-3%,
  Beijing under 1%). The Shanghai difference survives the long-bandwidth DM test; Beijing's does
  not (p = 0.10 to 0.30), so it is within noise.
- **Stratified-by-deviation.** On Hangzhou alone, CA-HGTN's advantage over GCN+GRU was larger on
  routine windows (6.9%) than on the top-20% high-deviation windows (-2.3%), opposite the design
  rationale in §3. **This did not generalize**: Shanghai (13.4% on high-deviation windows vs 11.1% on routine)
  and Beijing (8.5% vs 1.3%) show the reverse, with the advantage larger on high-deviation
  windows. Absolute error scale does not explain the split. The cause is unidentified. The mechanism story in §3 is the design intent, not a
  confirmed explanation of why the model performs well.
- **Gap to PB-GRU.** A separately published number for this benchmark family (PB-GRU) reports
  15-minute Hangzhou MAE 22.13 versus this project's 24.23 (extended; about 9.5% higher) and 26.74
  (standard). Preprocessing-protocol differences were investigated and judged too limited in scope
  to explain the gap (non-operating-hours handling, train/test date boundaries, aggregation
  resolution, missing-value handling). The likely remaining explanation is model scale (the
  project notes put PB-GRU at roughly 10-20x more capacity) and training regime (a fixed 350-epoch
  schedule versus early-stopped runs), not confirmed; matching both was estimated at roughly 600x
  more compute per seed and judged impractical (both figures are from `PAPER_PLAN_2.md`, not
  independently verified). A capacity-scaling check (GCN+GRU, hidden size 64 to 256, 15.3x
  parameters, 1 seed) bought 0.37 MAE. Temporal-component design bought more per parameter: the
  bidirectional-attention variant 27.98 vs 28.54 (-0.56, ~2x parameters, five-seed means), while
  the stacked variant (~3.4x) was worse than the single bidirectional layer (28.19 vs 27.98).
  Bidirectionality plus attention pooling helps; extra depth on top does not compound it. Even
  the best variant is still well short of PB-GRU's reported number, and CA-HGTN (full) still leads
  every GCN+GRU variant by a real margin on Hangzhou.
- **Diagnosed but unresolved:** at a subset of stations the prediction depends heavily on a single
  dominant correlation-graph neighbor, inheriting its demand pattern even where that does not
  match the station's own history. The paper reports two failed interventions without naming them;
  the saved station plots (`correlation-temperature=8.0`, `edge-dropout=0.15`) suggest they were
  the model's `correlation_temperature` and `edge_dropout` options. Confirm before citing this.
