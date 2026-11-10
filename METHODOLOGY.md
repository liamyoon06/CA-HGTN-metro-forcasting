# Methodology

This document describes the current, decided method: what it does and why, as reference
documentation. For the reasoning trail that led here -- including rejected alternatives,
bugs caught during development, and experiments that didn't work -- see `PAPER_PLAN.md`.

## 1. Problem formulation

Given a lookback window of `T=12` past 15-minute intervals of station-level inflow/outflow
counts across `N` stations, predict a full probability distribution over inflow/outflow
at each of the next `H=4` horizons (15/30/45/60 minutes ahead), not just a point estimate.
Ridership counts are non-negative, over-dispersed (variance exceeds the mean, especially
at high-traffic stations and during rush hours), and strongly periodic (time-of-day,
day-of-week). The method is designed around these three properties specifically.

## 2. Data and splits

LibCity-format datasets (HZMetro, SHMetro, Beijing subway), 15-minute aggregation. Splits
are date-anchored, not randomly shuffled, to avoid leakage between temporally adjacent
windows:

| | Train | Official val | Test | Internal fit | Internal val |
|---|---|---|---|---|---|
| HZMetro | 2019-01-01 to 01-18 | 01-19 to 01-20 | 01-21 to 01-25 | 01-01 to 01-11 | 01-12 to 01-18 |
| SHMetro | 2016-07-01 to 08-31 | 09-01 to 09-09 | 09-10 to 09-30 | 07-01 to 08-21 | 08-22 to 08-31 |

"Internal fit/val" is a strict subset of "train," used only for epoch-budget selection
(§5) -- the official validation period is 100% weekend for HZMetro and is never used for
tuning, only left available for other purposes. Internal validation periods are chosen to
mix weekday and weekend days specifically, so epoch-budget selection isn't tuned against
an unrepresentative validation distribution.

Three graph structures are available as spatial inputs:
- **Physical**: station adjacency from the transit network topology.
- **Correlation**: a directed, multi-lag cross-correlation graph, built from
  deseasonalized flow series (subtracting each station's own historical per-timeslot
  mean before computing correlation, to avoid shared rush-hour timing dominating every
  edge at lag 0).
- **Precomputed similarity/correlation**: the dataset's own `.rel`-file columns, which
  are exactly PVCGN's original graph construction (verified: edge counts and thresholds
  match PVCGN's published figures exactly). Available as an alternative spatial input;
  see §8 for how it compares to the in-project correlation graph.

## 3. Architecture (CA-HGTN)

1. **Input encoding**: raw counts are log1p-compressed, then linearly projected to
   `hidden_dim` (default 64), added to a learned time-of-day/day-of-week embedding.
2. **Spatial layers** (2, stacked): multi-graph attention over stations at each lookback
   timestep. Up to three graphs are fused via a learned, softmax-normalized gate:
   physical (row-normalized, cached since it's fixed for an entire training run),
   correlation (same), and an adaptive graph (`softmax(relu(E1 @ E2^T))` over learned
   per-station embeddings, recomputed every call since it depends on trainable
   parameters). The fused graph enters standard scaled-dot-product attention as an
   additive log-bias, not a multiplicative mask -- attention's own softmax renormalizes
   regardless of the bias's row-sum. `graph_mode` selects which subset of the three
   graphs is used (`full`, `physical_only`, `correlation_only`, `adaptive_only`,
   `no_physical`), supporting the ablations in §7.
   - *Optional, off by default*: `edge_dropout` randomly zeros a fraction of each graph's
     edges, freshly every training step (train mode only), adapting DropEdge (Rong et
     al., ICLR 2020).
3. **Temporal fusion**: a Transformer encoder (2 layers, 4 heads) and a bidirectional
   GRU process the lookback sequence in parallel; their outputs are combined via a
   learned gate, then attention-pooled across the lookback window.
4. **Output head — seasonal-residual decomposition**: rather than predicting ridership
   from scratch, the model predicts a *bounded multiplicative correction* around each
   station's own historical mean for that specific horizon's time-of-day slot:
   ```
   prediction = (seasonal_baseline + 1.0) * exp(clamp(correction, -3, 3))
   ```
   The correction head is zero-initialized, so training starts exactly at the seasonal
   baseline and only deviates as far as the data justifies. The `+1.0` floor avoids
   multiplying by exactly zero for genuinely zero-traffic (station, timeslot) cells; the
   `[-3, 3]` clamp bounds the multiplicative correction to roughly `[0.05x, 20x]` of the
   seasonal baseline, preventing runaway predictions early in training.
5. **Distributional head**: `correction` parameterizes `mu` (the Negative Binomial mean)
   via the mechanism above; a separate small head predicts `r` (dispersion), also via a
   correction around a fixed reference initialization, giving `Var = mu + mu^2/r` --
   larger `r` means lower over-dispersion, approaching Poisson as `r -> infinity`.

## 4. Loss function

**Negative Binomial negative log-likelihood**, the primary training signal for `mu` and
`r` jointly, is not used alone. Plain NB-NLL has a failure mode: the model can lower its
loss by shrinking `r` (declaring high uncertainty) for a hard-to-predict cell, rather than
improving `mu` -- verified numerically: shrinking `r` from a healthy value to a small one,
for a fixed, badly-wrong `mu`, cuts the loss substantially without ever correcting the
mean. Two complementary, combinable mechanisms address this:

- **Auxiliary point loss** (`aux_weight=0.15` by default): a target-magnitude-weighted
  Huber loss on `mu` directly, added to the NB-NLL. Its gradient doesn't depend on `r`,
  so it keeps pressure on `mu` regardless of what `r` does.
- **Beta-NLL reweighting** (`beta`, off by default at 0): adapts Seitzer et al. (ICLR
  2022) from their Gaussian heteroscedastic case to this Negative Binomial one --
  multiplies each cell's NLL contribution by `(r.detach() / r_reference) ** (-beta)`
  (`r_reference=10.0`, matching this project's own dispersion-head initialization),
  up-weighting cells where `r` has shrunk. `r.detach()` prevents the reweighting itself
  from creating a new feedback loop into `r`. Verified to neutralize the specific
  escape-hatch scenario above at `beta` roughly in `[0.75, 1.0]`. See §8 for how it
  performs in practice, isolated from the auxiliary term.

## 5. Training protocol

Two-phase, specifically to avoid tuning the epoch budget on data used in the final model:

1. **Phase 1 (epoch-budget selection)**: train on `internal_fit`, early-stop against
   `internal_val`. Early stopping requires `min_epochs_before_stopping=25` epochs before
   it can trigger, and a relative improvement of at least `min_delta_rel=0.15%` to reset
   patience (`patience=15` epochs without such an improvement before stopping) --
   deliberately conservative given this dataset's small size (order of 1,000-4,500
   overlapping training windows depending on city) makes naive early stopping prone to
   noise-driven premature stops.
2. **Phase 2 (final refit)**: retrain from scratch on the full official `train` period,
   for exactly the epoch count selected in Phase 1, with no further validation signal
   (there is none left to use without leakage).

`AdamW`, `weight_decay=1e-3`, linear LR warmup (5 epochs) then cosine decay, gradient
clipping at norm 5.0. Repeated across `N_SEEDS=5` independent seeds; each seed
independently repeats both phases (a different seed can genuinely need a different epoch
budget -- sharing budgets across seeds would compromise the independence multi-seed
evaluation exists to measure).

## 5b. A deliberate exception to the above, clearly separated

`diagnostic_train_with_test_trajectory` (`train.py`) breaks the "never touch the test set
during training" rule above, on purpose, for one specific question: does long, unstopped
training on this dataset actually overfit? Trains for a long fixed budget (300 epochs,
approaching PB-GRU's published 350) with early stopping disabled, checking test MAE
periodically to trace its trajectory. This is legitimate only as an observational
diagnostic -- its output must never be used to select an epoch count, checkpoint, or
hyperparameter for anything reported as an official result. Kept structurally separate:
a distinctly-named function, a separate output file, deliberately excluded from the
aggregation notebook.

## 6. Evaluation

- **Point metrics**: MAE, RMSE, MAPE, WMAPE.
- **Distributional metrics**: coverage@90/coverage@50 (do X% nominal intervals actually
  contain X% of observations), NB negative log-likelihood, and NASS (a skill-score
  framework adapting Murphy 1996 / Nash-Sutcliffe 1970 / Gneiting & Raftery 2007, using a
  model-endogenous reference floor from fitted dispersion rather than a fixed external
  baseline).
- **Ensembling**: predictions are averaged across the 5 seeds (exact for point accuracy;
  dispersion is combined via the law of total variance, flagged as an approximation, not
  an exact NB mixture).
- **Significance testing**: Diebold-Mariano (HAC-corrected for multi-step-ahead
  autocorrelation, with the Harvey-Leybourne-Newbold small-sample correction) and
  Wilcoxon signed-rank, both validated against known statistical properties before use
  (measured false-positive rate under no true difference, not assumed correct).
- **Relative improvement**: bootstrap confidence intervals (resampling test windows with
  replacement) for percentage improvement over a specific, justified baseline -- not a
  bare point percentage, and not a delta-method approximation over only 5 seeds.
- **Stratified-by-deviation analysis**: metrics computed separately on the top 20% of
  test windows by deviation from the seasonal baseline, versus the rest -- checking
  whether a model's advantage concentrates where graph-based information-sharing should
  matter most (anomalous conditions) versus routine conditions. See §8 for the result.

## 7. Baselines

- **GRU-only**: the same seasonal-residual output mechanism and training protocol, no
  spatial structure at all.
- **GCN+GRU**: a standard Kipf & Welling-style graph convolution + GRU, same output
  mechanism and protocol -- reproduced under this project's own pipeline specifically to
  get an apples-to-apples comparison point, since cross-paper reported numbers in this
  literature were found to disagree substantially for the same cited model (see §8).
  An optional variant (`temporal_mode="bidirectional_attention"`, superseding the earlier
  `bidirectional_attention=True` flag which is kept as a backward-compatible alias)
  upgrades only the temporal component -- bidirectional GRU with learned attention
  pooling over the lookback window, replacing the default's plain unidirectional GRU
  with last-hidden-state pooling -- holding capacity, graph, and everything else fixed,
  to isolate whether temporal-component sophistication (not model size) explains part
  of the gap discussed in §8. Costs ~2x parameters at the default hidden_dim, against
  ~15x for the capacity-scaling check in the same section. A further variant
  (`temporal_mode="stacked_bidir_uni_attention"`, ~3.4x parameters) stacks a
  unidirectional GRU layer after the bidirectional one before pooling, a reasonable
  interpretation (not a verified exact replication -- source code unavailable) of
  PB-GRU's own description of its temporal component.
- **Architectural ablations** (via `graph_mode` and `head_type`): single-graph
  (physical topology only), no-physical (correlation + adaptive, no physical anchor),
  and point-head (removes the distributional head and auxiliary-loss structure entirely,
  training on the point loss alone).

## 8. Current results and open findings

Full numbers are in `PAPER_PLAN.md`; summarized here:

- CA-HGTN (full) significantly outperforms every baseline and ablation above (DM and
  Wilcoxon, p < 0.001, all 4 horizons) on HZMetro, 5 seeds. ~5.8% improvement (95% CI
  [4.8%, 6.7%]) over the reproduced GCN+GRU baseline specifically -- the most trustworthy
  single number available, for the reason given above.
- Swapping in the dataset's own precomputed (PVCGN-style) graphs, in place of this
  project's own correlation construction, made results worse (27.6 vs 26.7 MAE) despite
  being verified to be exactly what the compared literature uses. Likely explanation:
  those graphs were tuned for a different downstream architecture (PVCGN/PB-GRU use
  different normalization per graph type -- Laplace for physical, random-walk for
  similarity -- while this architecture applies uniform row-normalization to whichever
  graph fills that slot) -- matching graph *data* isn't the same as matching graph
  *usage*.
- Beta-NLL reweighting, tested in isolation (replacing rather than supplementing the
  auxiliary loss term), performed worse on accuracy, stability, *and* calibration
  simultaneously. The escape-hatch-specific mechanism still appears to work as verified;
  the auxiliary term evidently provides broader training-stability value beyond that one
  failure mode. Combining both (not yet tested) remains open.
- The stratified-by-deviation analysis found the model's advantage over baselines
  concentrated in *routine*, not anomalous, conditions -- the opposite of the design
  rationale in §3 (graph-based information-sharing should matter most during anomalies,
  when a model with no such mechanism has no way to compensate). Held across every
  CA-HGTN variant tested, not one noisy run. The mechanism story in §3 should be treated
  as the design intent, not a confirmed explanation of why the model performs well.
- A separate published number for this same benchmark family (Parallel-MGCN/PB-GRU)
  reports substantially lower error (~22 MAE vs. this project's 26.7). Investigated and
  ruled out as a preprocessing-protocol difference (checked: non-operating-hours
  handling, train/test date boundaries, aggregation resolution, and missing-value
  handling are all either identical across papers or too limited in scope to explain a
  20%+ gap). The likely remaining explanation is model scale (that work uses 10-20x more
  parameters) and training regime (a fixed 350-epoch schedule versus this project's
  early-stopped ~50-90) -- not yet confirmed, and literally matching those settings was
  estimated to cost roughly 600x more compute per seed, which was judged impractical.
  A capacity-scaling check (same baseline architecture, larger `hidden_dim`) showed a
  small, monotonic trend in the expected direction, but not a large one at the scales
  tested -- a 15.3x parameter increase (hidden_dim 64->256) bought 0.37 MAE (1 seed
  only, so that specific point estimate carries real uncertainty). A narrower check
  isolating temporal-component sophistication instead of scale (`bidirectional_attention`
  variant, §7) -- both compared as proper 5-seed means -- bought 0.56 MAE for only ~2x
  parameters, a more parameter-efficient result than the capacity check, and with lower
  cross-seed variance too (+-0.20 vs +-0.34). A further check stacking a second,
  unidirectional GRU layer on top (`stacked_bidir_uni_attention`, ~3.4x parameters)
  performed WORSE than the single bidirectional layer (28.19+-0.17 vs 27.98+-0.20 at
  15min, consistent across horizons), though still better than the original. The precise
  finding, then: bidirectionality + attention pooling helps meaningfully; additional
  depth on top does not compound that effect and may mildly hurt. Take the overall
  picture as evidence for temporal-component design as one real, contributing factor,
  not as an explanation of most of the gap -- even the best variant tested is still ~6
  points from PB-GRU's reported number, and CA-HGTN (full) still leads every GCN+GRU
  variant by a real margin.
