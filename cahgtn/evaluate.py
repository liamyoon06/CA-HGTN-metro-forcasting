"""
cahgtn/evaluate.py
================================================================
Standard point metrics (MAE, RMSE, MAPE, WMAPE) plus two things
deterministic baselines in this literature cannot report at all,
because they don't produce a distribution:

1. Calibration diagnostics: empirical coverage of the model's own
   50%/90% Negative-Binomial prediction intervals. A well-calibrated
   model should cover ~50%/~90% of true values in those intervals
   respectively; point-forecast models have no equivalent check.

2. Noise-Adjusted Skill Score (NASS): a new evaluation metric for
   this domain. For count data, MAPE is dominated by an irreducible
   sampling floor at low counts (see PLAN.md / paper draft for the
   derivation). NASS uses the model's own fitted dispersion to
   estimate that floor per cell, then reports how much of the *gap
   between a naive baseline and the noise floor* the model actually
   closes:

       skill_i = (e_naive_i - e_model_i) / (e_naive_i - floor_i)

   averaged over cells where the naive baseline has room to improve
   at all (e_naive_i > floor_i). NASS = 1 means the model is doing as
   well as the noise floor allows; NASS = 0 means it's no better than
   the naive baseline; NASS < 0 means it's worse.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import nbinom


def ensemble_predictions(seed_preds: list[dict]) -> dict:
    """Averages predictions across independently-trained seeds -- a cheap,
    well-established technique (no retraining required) that typically both
    improves point accuracy and reduces variance relative to any single
    seed. Directly motivated by what the 5-seed HZMetro run showed: `point
    head`'s seed-to-seed MAE std (+-5.55) was far larger than `full`'s
    (+-0.71), meaning individual seeds for that variant are unreliable on
    their own -- ensembling is a natural, immediate mitigation using
    predictions you've already computed.

    seed_preds: list of the `preds` dict returned by predict_all for each
    seed (all for the SAME variant, same test set). Each dict has either
    {'mu': ..., 'r': ...} (NB head) or {'pred': ...} (point head).

    For point accuracy (MAE/RMSE/MAPE/WMAPE), averaging `mu` (or `pred`)
    across seeds is a fully valid ensemble of point forecasts -- no
    approximation involved.

    For calibration (coverage, NASS), ensembling `r` is NOT simply
    averaging r across seeds: the true ensemble predictive distribution is
    a MIXTURE of 5 Negative Binomials, which doesn't have a closed-form NB
    itself. What's returned here for 'r' is an approximation via the law of
    total variance: Var(mixture) = E[Var(seed)] + Var[E[seed]], i.e. the
    average of each seed's own predicted variance, PLUS the variance of the
    seeds' mu predictions around their mean (capturing genuine model
    disagreement across seeds, which a single seed's r cannot). That total
    variance is then converted back to an approximating single-NB r via
    r = mu^2 / (variance - mu) when variance > mu (else the mixture is
    under-dispersed relative to Poisson at that point, r is set very large).
    This is a reasonable, defensible approximation for reporting purposes,
    not an exact treatment -- flag it as such if used in the paper.
    """
    if "mu" in seed_preds[0]:
        mus = np.stack([p["mu"] for p in seed_preds], axis=0)   # (S, ...)
        rs = np.stack([p["r"] for p in seed_preds], axis=0)      # (S, ...)
        ens_mu = mus.mean(axis=0)

        within_seed_var = (mus + (mus ** 2) / np.clip(rs, 1e-6, None)).mean(axis=0)
        across_seed_var = mus.var(axis=0)
        total_var = within_seed_var + across_seed_var

        excess = total_var - ens_mu
        ens_r = np.where(excess > 1e-6, (ens_mu ** 2) / np.clip(excess, 1e-6, None), 1e6)
        return {"mu": ens_mu, "r": ens_r}
    else:
        preds = np.stack([p["pred"] for p in seed_preds], axis=0)
        return {"pred": preds.mean(axis=0)}


def high_deviation_mask(trues: np.ndarray, naive_preds: np.ndarray, top_fraction: float = 0.2) -> np.ndarray:
    """Returns a boolean mask over windows (axis 0) identifying the
    `top_fraction` with the largest relative deviation from the seasonal
    baseline -- i.e. the most anomalous conditions in the test set.

    Motivation: a graph-aware model's real structural advantage over a
    baseline with no such mechanism should show up specifically here --
    on ROUTINE windows (small deviation from the historical time-of-day
    norm), the seasonal-residual mechanism already does most of the work
    for every variant equally, leaving little room for architecture to
    matter. On ANOMALOUS windows, a model that can borrow signal from
    correlated stations' current anomalies has a real, checkable
    structural reason to do better than one that can't. This is a
    principled slice motivated by the architecture's actual mechanism,
    not a post-hoc search for a subset where the numbers look better --
    report the routine-window numbers alongside it, not instead of it.

    trues, naive_preds: (num_windows, horizon, N, 2), same shape/alignment
    predict_all and the saved naive_test array already use.
    """
    diff = np.abs(trues - naive_preds)
    rel_deviation = diff / (naive_preds + 1.0)
    per_window_deviation = rel_deviation.reshape(rel_deviation.shape[0], -1).mean(axis=1)
    threshold = np.quantile(per_window_deviation, 1 - top_fraction)
    return per_window_deviation >= threshold


def apply_window_mask(preds: dict, trues: np.ndarray, mask: np.ndarray,
                       naive_preds: np.ndarray | None = None):
    """Restricts preds/trues/naive_preds to windows where mask is True, for
    use with per_horizon_report -- e.g. per_horizon_report(*apply_window_mask(...))."""
    if "mu" in preds:
        preds_masked = {"mu": preds["mu"][mask], "r": preds["r"][mask]}
    else:
        preds_masked = {"pred": preds["pred"][mask]}
    trues_masked = trues[mask]
    naive_masked = naive_preds[mask] if naive_preds is not None else None
    return preds_masked, trues_masked, naive_masked


def exclude_stations_mask(num_stations: int, exclude: list) -> np.ndarray:
    """Boolean mask over stations (True = keep) with the given 0-indexed
    station indices excluded. Small helper for apply_station_mask."""
    mask = np.ones(num_stations, dtype=bool)
    mask[list(exclude)] = False
    return mask


def include_only_stations_mask(num_stations: int, include: list) -> np.ndarray:
    """Boolean mask over stations (True = keep) including ONLY the given
    0-indexed station indices -- the symmetric complement of
    exclude_stations_mask, for isolating a specific subset (e.g. stations
    already flagged as structurally at-risk) rather than excluding one."""
    mask = np.zeros(num_stations, dtype=bool)
    mask[list(include)] = True
    return mask


def apply_station_mask(preds: dict, trues: np.ndarray, station_mask: np.ndarray,
                        naive_preds: np.ndarray | None = None):
    """Restricts preds/trues/naive_preds to stations where station_mask is
    True (axis 2 of (num_windows, horizon, num_stations, 2)), for use with
    per_horizon_report -- e.g. per_horizon_report(*apply_station_mask(...)).

    Same principle this project already applies to window-level slicing
    (see high_deviation_mask): report the full-network numbers ALONGSIDE
    any station subset, never in place of them. A subset chosen because
    specific stations look bad in isolation is a post-hoc search for a
    better-looking number, not a legitimate analysis slice -- a legitimate
    one has a reason that doesn't depend on having already seen how those
    stations score (a known data issue, a station category decided in
    advance, an out-of-network validation split)."""
    if "mu" in preds:
        preds_masked = {"mu": preds["mu"][:, :, station_mask, :], "r": preds["r"][:, :, station_mask, :]}
    else:
        preds_masked = {"pred": preds["pred"][:, :, station_mask, :]}
    trues_masked = trues[:, :, station_mask, :]
    naive_masked = naive_preds[:, :, station_mask, :] if naive_preds is not None else None
    return preds_masked, trues_masked, naive_masked


def point_metrics(pred: np.ndarray, true: np.ndarray, mape_thresh: float = 1.0) -> dict:
    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mask = np.abs(true) >= mape_thresh
    mape = float(np.mean(np.abs(diff[mask]) / np.abs(true[mask])) * 100) if mask.sum() else float("nan")
    wmape = float(np.sum(np.abs(diff)) / np.sum(np.abs(true)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WMAPE": wmape}


def _nb_scipy_params(mu: np.ndarray, r: np.ndarray):
    """Convert (mu, r) -> scipy.stats.nbinom's (n, p) parameterization.
    scipy nbinom: mean = n*(1-p)/p, matching our r*(1-p)/p with n = r."""
    p = r / (r + mu + 1e-8)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return r, p


def find_calibration_scale(mu: np.ndarray, r: np.ndarray, true: np.ndarray,
                            target_coverage: float = 0.90, tol: float = 0.002,
                            max_iter: int = 60, c_bounds: tuple = (0.01, 100.0)) -> float:
    """Post-hoc recalibration: finds a scalar c such that r_calibrated = r / c
    achieves target_coverage on the given (mu, r, true) data, via binary
    search. Verified numerically (not assumed) that coverage is monotonically
    increasing in c: larger c -> smaller effective r -> wider NB distribution
    -> higher coverage -- see PAPER_PLAN.md for the check.

    Adapts the standard post-hoc calibration idea (e.g. temperature scaling
    in classification) to NB regression intervals: rather than changing
    training to fix miscalibration, rescale the already-fitted dispersion
    after the fact, using a held-out set that is NOT the final test set the
    reported numbers come from. Critically, this does not touch mu at all --
    point-accuracy metrics (MAE, RMSE, WMAPE) are mathematically unaffected;
    only calibration-related metrics (coverage, NB-NLL) change. This means
    it cannot undo an accuracy gain while fixing calibration -- the two are
    structurally independent under this correction.

    Motivated directly by this project's own finding: extended training (the
    epoch-floor result) improved MAE by 1.69 points while coverage@90
    dropped from ~0.874 to ~0.846 -- consistent with r learning to reflect
    how well mu fits the TRAIN set specifically (which keeps improving with
    more epochs, driven substantially by the aux point-loss term, which is
    r-independent) without that confidence fully transferring to test data,
    where mu's accuracy is somewhat lower. This corrects for the resulting
    overconfidence after training, rather than trying to prevent it during
    training (a harder, less-understood intervention)."""
    level_key = f"coverage@{int(target_coverage * 100)}"
    lo, hi = c_bounds
    cov_lo = nb_calibration(mu, r / lo, true, level=target_coverage)[level_key]
    cov_hi = nb_calibration(mu, r / hi, true, level=target_coverage)[level_key]
    if not (cov_lo <= target_coverage <= cov_hi):
        raise ValueError(
            f"target_coverage={target_coverage} not reachable within c_bounds={c_bounds} "
            f"(coverage ranges [{cov_lo:.3f}, {cov_hi:.3f}] over that range) -- widen c_bounds."
        )
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        cov_mid = nb_calibration(mu, r / mid, true, level=target_coverage)[level_key]
        if abs(cov_mid - target_coverage) < tol:
            return mid
        if cov_mid < target_coverage:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def apply_calibration_scale(r: np.ndarray, c: float) -> np.ndarray:
    """r_calibrated = r / c, from find_calibration_scale. Named separately
    from the search itself so it's clear at the call site that this is the
    (trivial) application step, applied to a DIFFERENT dataset (e.g. test)
    than the one c was fit on (e.g. internal_val) -- never fit and apply
    the scale on the same data."""
    return r / c


def nb_calibration(mu: np.ndarray, r: np.ndarray, true: np.ndarray, level: float = 0.9) -> dict:
    """Empirical coverage of the [alpha/2, 1-alpha/2] NB interval, mean NLL, and
    mean interval width (sharpness).

    Coverage alone is gameable: an infinitely wide interval trivially achieves
    100% coverage while conveying zero information. The standard framework for
    probabilistic forecasts (Gneiting et al.) is to maximize sharpness SUBJECT
    TO calibration -- the best interval is the narrowest one that still
    achieves correct coverage, not just any interval that does. Width is
    reported alongside coverage specifically so post-hoc recalibration (or any
    other calibration fix) can be checked for whether it's a modest, honest
    correction or a degenerate one that trades away all informativeness to hit
    the target."""
    n, p = _nb_scipy_params(mu, r)
    alpha = 1 - level
    lo = nbinom.ppf(alpha / 2, n, p)
    hi = nbinom.ppf(1 - alpha / 2, n, p)
    covered = ((true >= lo) & (true <= hi)).mean()
    width_mean = float((hi - lo).mean())

    log_r = np.log(r + 1e-8)
    log_mu = np.log(mu + 1e-8)
    log_r_mu = np.log(r + mu + 1e-8)
    from scipy.special import gammaln
    ll = (gammaln(true + r) - gammaln(r) - gammaln(true + 1)
          + r * (log_r - log_r_mu) + true * (log_mu - log_r_mu))
    return {f"coverage@{int(level*100)}": float(covered), "NB_NLL": float(-ll.mean()),
            f"width@{int(level*100)}": width_mean}


def noise_adjusted_skill_score(pred_mu: np.ndarray, r: np.ndarray, true: np.ndarray,
                                naive_pred: np.ndarray, min_gap: float = 1e-3) -> dict:
    """See module docstring. Returns headline NASS plus supporting diagnostics.

    NOTE ON FORMULATION: an earlier version of this function averaged a
    *per-cell* ratio (e_naive_i - e_model_i) / (e_naive_i - floor_i). That is
    numerically unstable in exactly the way pointwise MAPE is: whenever a
    cell's naive error happens to sit only slightly above its estimated noise
    floor, the denominator is tiny and the ratio blows up (we measured raw
    per-cell values into the double digits on synthetic test data, which is
    meaningless for a score meant to sit roughly in [0, 1]). This version
    instead aggregates numerator and denominator as SUMS before dividing --
    the same "weighted/aggregate over pointwise" fix used for WMAPE vs MAPE
    -- which is stable because a handful of near-zero individual gaps can no
    longer dominate the result.
    """
    sigma = np.sqrt(pred_mu + (pred_mu ** 2) / np.clip(r, 1e-6, None))
    floor = sigma  # absolute-scale irreducible noise estimate

    e_model = np.abs(pred_mu - true)
    e_naive = np.abs(naive_pred - true)

    # Only cells where the naive baseline has genuine room to improve at all.
    gap_naive = np.clip(e_naive - floor, 0.0, None)
    valid = gap_naive > min_gap

    # Excess error beyond the estimated floor, clipped at 0 (a model doing
    # better than its own estimated floor on a given cell is noise/luck, not
    # skill, and shouldn't be allowed to offset genuine misses elsewhere).
    excess_model = np.clip(e_model[valid] - floor[valid], 0.0, None)
    excess_naive = gap_naive[valid]

    denom = excess_naive.sum()
    nass = float(1.0 - excess_model.sum() / denom) if denom > 0 else float("nan")

    return {
        "NASS": nass,
        "pct_cells_with_headroom": float(valid.mean() * 100),
        "mean_estimated_floor": float(floor.mean()),
        "mean_naive_error": float(e_naive.mean()),
        "mean_model_error": float(e_model.mean()),
    }


def per_horizon_report(preds: dict, trues: np.ndarray, naive_preds: np.ndarray | None = None) -> list[dict]:
    """preds: for NB models, dict with 'mu' (H,N,2) and 'r' (H,N,2) arrays (already
    concatenated over test windows on axis 0 -> shape (num_windows, H, N, 2)); for
    point models, dict with 'pred' of the same shape. trues: same shape as pred/mu."""
    horizon = trues.shape[1]
    reports = []
    is_nb = "mu" in preds
    for h in range(horizon):
        true_h = trues[:, h].reshape(-1)
        if is_nb:
            mu_h = preds["mu"][:, h].reshape(-1)
            r_h = preds["r"][:, h].reshape(-1)
            rep = point_metrics(mu_h, true_h)
            rep.update(nb_calibration(mu_h, r_h, true_h, level=0.9))
            rep.update(nb_calibration(mu_h, r_h, true_h, level=0.5))
            if naive_preds is not None:
                rep.update(noise_adjusted_skill_score(mu_h, r_h, true_h, naive_preds[:, h].reshape(-1)))
        else:
            pred_h = preds["pred"][:, h].reshape(-1)
            rep = point_metrics(pred_h, true_h)
        reports.append(rep)
    return reports


def print_report(city: str, model_name: str, reports: list[dict]):
    labels = ["15min", "30min", "45min", "60min"][: len(reports)]
    print(f"\n=== {city} / {model_name} ===")
    for label, r in zip(labels, reports):
        line = f"  {label:>6}  MAE {r['MAE']:6.2f}  RMSE {r['RMSE']:6.2f}  MAPE {r['MAPE']:6.2f}%  WMAPE {r['WMAPE']:6.2f}%"
        if "NASS" in r:
            line += f"  NASS {r['NASS']:.3f}  cov@90 {r.get('coverage@90', float('nan')):.3f}  cov@50 {r.get('coverage@50', float('nan')):.3f}"
        print(line)


def plot_station_forecast(ens: dict, trues: np.ndarray, station_idx: int, horizon_idx: int = 0,
                           flow_idx: int = 0, station_label: str | None = None,
                           uncertainty_level: float = 0.90, save_path: str | None = None):
    """Actual vs. predicted ridership for ONE station across all test windows,
    at a single horizon and flow direction.

    ens: dict with 'mu'/'r' (NB variants -- shows a shaded predictive interval,
         using the SAME NB parameterization as nb_calibration via the shared
         _nb_scipy_params helper, not a separate reimplementation) or 'pred'
         (point variants -- no interval to show), each shape
         (num_windows, horizon, num_stations, 2).
    trues: same shape.
    station_idx: 0-indexed. NOTE: this project's .geo files have no station
         names (just empty coordinate placeholders, verified directly) --
         station_label lets you supply your own if you have external
         knowledge of what each index corresponds to; otherwise defaults to
         "Station {station_idx}". Don't assume a name exists that isn't in
         the data.
    horizon_idx: 0=15min, 1=30min, 2=45min, 3=60min (this project's standard).
    flow_idx: 0=inflow, 1=outflow.
    uncertainty_level: width of the shown interval (NB variants only).
    save_path: if given, saves the figure (e.g. into /mnt/user-data/outputs).

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    is_nb = "mu" in ens
    true_series = trues[:, horizon_idx, station_idx, flow_idx]
    x = np.arange(true_series.shape[0])

    fig, ax = plt.subplots(figsize=(11, 4.5))

    if is_nb:
        mu_series = ens["mu"][:, horizon_idx, station_idx, flow_idx]
        r_series = ens["r"][:, horizon_idx, station_idx, flow_idx]
        n_param, p_param = _nb_scipy_params(mu_series, r_series)
        alpha = 1 - uncertainty_level
        lo = nbinom.ppf(alpha / 2, n_param, p_param)
        hi = nbinom.ppf(1 - alpha / 2, n_param, p_param)
        ax.fill_between(x, lo, hi, alpha=0.2, color="tab:blue",
                         label=f"{int(uncertainty_level * 100)}% predictive interval")
        pred_series = mu_series
        pred_label = "Predicted mean ($\\mu$)"
    else:
        pred_series = ens["pred"][:, horizon_idx, station_idx, flow_idx]
        pred_label = "Predicted"

    ax.plot(x, true_series, color="black", linewidth=1.3, label="Actual")
    ax.plot(x, pred_series, color="tab:blue", linewidth=1.3, linestyle="--", label=pred_label)

    mae = float(np.mean(np.abs(pred_series - true_series)))
    flow_name = "Inflow" if flow_idx == 0 else "Outflow"
    horizon_labels = ["15min", "30min", "45min", "60min"]
    horizon_str = horizon_labels[horizon_idx] if horizon_idx < len(horizon_labels) else f"horizon {horizon_idx}"
    label = station_label or f"Station {station_idx}"

    ax.set_title(f"{label} -- {flow_name}, {horizon_str} horizon (MAE={mae:.2f})")
    ax.set_xlabel("Test window index (sequence order, not wall-clock time -- not tracked in this pipeline)")
    ax.set_ylabel("Ridership count")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)

    return fig
