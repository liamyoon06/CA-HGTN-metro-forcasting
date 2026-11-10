"""
cahgtn/data_pipeline.py
================================================================
Loads raw LibCity-format .dyna/.rel files for HZMETRO / SHMETRO /
BEIJING_SUBWAY_15MIN and builds protocol-correct train/val/test
tensors.

Key methodological fixes vs. the original project + vs. common
practice in the literature:

1. TRAIN/TEST uses the published, date-anchored split (comparable
   to PVCGN and follow-ups) rather than a fractional split.
2. The INTERNAL validation set used for early stopping / hyper-
   parameter selection is carved from *within* the training window
   and deliberately mixes weekday + weekend days. The official
   published validation block for HZMetro (Jan 19-20, 2019) is
   100% weekend while the official test block (Jan 21-25) is 100%
   weekday -- using the official val block for model selection
   silently tunes toward the wrong distribution. We still evaluate
   final numbers on the official test block for comparability.
3. The StandardScaler-equivalent normalization (if used at all) is
   fit on TRAIN ONLY, never on the full dataset.
4. Builds three graph views for the multi-graph encoder: physical
   topology (from .rel), lag-correlation graph (computed on TRAIN
   ONLY, so no test leakage), and leaves room for a learned
   adaptive graph (built inside the model itself).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

LOOKBACK = 12   # 3 hours of 15-min bins
HORIZON = 4     # next 15 / 30 / 45 / 60 minutes
DAILY_LAG = 73  # steps in one HZMetro/SHMetro operating day; recomputed per-city below

CITY_FILES = {
    "Hangzhou": "HZMETRO",
    "Shanghai": "SHMETRO",
    "Beijing": "BEIJING_SUBWAY_15MIN",
}

# Published (or best-available) date splits. HZMetro/SHMetro splits follow
# PVCGN and its follow-ups. Beijing has no widely-cited split; we fall back
# to a chronological 70/15/15 split and flag this in reports.
DATE_SPLITS = {
    "Hangzhou": {
        "train": ("2019-01-01", "2019-01-18"),
        "official_val": ("2019-01-19", "2019-01-20"),   # 100% weekend -- do not use for tuning
        "test": ("2019-01-21", "2019-01-25"),
        "internal_fit": ("2019-01-01", "2019-01-11"),     # for hyperparameter tuning
        "internal_val": ("2019-01-12", "2019-01-18"),     # mixes weekend (12-13) + weekday (14-18)
    },
    "Shanghai": {
        "train": ("2016-07-01", "2016-08-31"),
        "official_val": ("2016-09-01", "2016-09-09"),
        "test": ("2016-09-10", "2016-09-30"),
        "internal_fit": ("2016-07-01", "2016-08-21"),
        "internal_val": ("2016-08-22", "2016-08-31"),     # includes multiple weekday+weekend cycles
    },
    "Beijing": None,  # handled with fractional fallback
}


@dataclass
class CityTensors:
    city: str
    flow: np.ndarray                 # (T, N, 2) raw inflow/outflow counts
    time_index: pd.DatetimeIndex
    adjacency_physical: torch.Tensor  # (N, N)
    adjacency_corr: torch.Tensor      # (N, N), OUR directed multi-lag correlation graph, TRAIN only
    lag_matrix: np.ndarray            # (N, N) best-lag (in steps) behind each adjacency_corr edge, diagnostic only
    num_stations: int
    slot_id: np.ndarray                # (T,) 15-min slot-of-day index
    n_slots: int
    daily_lag: int
    idx: dict = field(default_factory=dict)  # named index arrays: train/test/internal_fit/internal_val
    adjacency_similarity_precomputed: torch.Tensor | None = None    # (N,N), from the .rel file's own "similarity" column
    adjacency_correlation_precomputed: torch.Tensor | None = None   # (N,N), from the .rel file's own "correlation" column


def _date_mask(index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    return np.asarray((index >= start_ts) & (index < end_ts))


def _build_physical_adjacency(rel: pd.DataFrame, n_stations: int) -> np.ndarray:
    A = np.zeros((n_stations, n_stations), dtype=np.float32)
    conn_col = "connection" if "connection" in rel.columns else None
    for row in rel.itertuples():
        i, j = int(row.origin_id), int(row.destination_id)
        if conn_col is not None and getattr(row, conn_col) != 1:
            continue
        if 0 <= i < n_stations and 0 <= j < n_stations:
            A[i, j] = 1.0
    return A


def _build_precomputed_edge_adjacency(rel: pd.DataFrame, n_stations: int, column: str) -> np.ndarray | None:
    """Extracts a precomputed edge-weight column (e.g. "similarity" or
    "correlation") from the .rel file directly, if present, as an (N,N)
    matrix. Returns None if the column isn't in this file.

    Motivation: the standard LibCity-format HZMetro/SHMetro .rel files (the
    same files this project's own data_pipeline.py reads) ship with
    precomputed "similarity" and "correlation" edge weights alongside
    "connection" (physical adjacency) -- and per a direct read of the
    PVCGN paper (arXiv 2001.04889) and multiple papers that build on it
    (GCN-SUBLSTM, PB-GRU/Parallel-MGCN, TSTA-GCN, HM-PVCGN), "physical,
    similarity, and correlation graphs" are literally the three standard
    virtual-graph inputs this entire benchmark family uses. This project's
    own `_build_lagged_correlation_adjacency` builds a DIFFERENT, from-
    scratch correlation graph (directed, deseasonalized, multi-lag) rather
    than using these precomputed columns -- a legitimate, more targeted
    construction on its own merits, but not what the compared literature
    is actually using. Exposed here as an additional, opt-in option
    (existing behavior via `adjacency_corr` is completely unchanged) so
    the standard-graph version can be tried directly by passing one of
    these into the same `adj_corr` parameter slot that CAHGTN/MultiGraphAttention
    already accept, with NO changes needed to model.py or train.py at all.
    """
    if column not in rel.columns:
        return None
    A = np.zeros((n_stations, n_stations), dtype=np.float32)
    for row in rel.itertuples():
        i, j = int(row.origin_id), int(row.destination_id)
        if 0 <= i < n_stations and 0 <= j < n_stations:
            A[i, j] = float(getattr(row, column))
    return A


def _build_lagged_correlation_adjacency(flow_train: np.ndarray, slot_id_train: np.ndarray, n_slots: int,
                                         max_lag: int = 8, top_k: int = 8):
    """Directed lead-lag graph: edge (source=i -> target=j) with weight equal to
    the strongest positive correlation between station i's flow at time t and
    station j's flow at time t+lag, over lag in [0, max_lag] (a 15-min-step
    lag range up to max_lag*15 minutes), computed on TRAIN data only.

    IMPORTANT: correlation is computed on each station's DESEASONALIZED
    residual (flow minus its own time-of-day mean), not raw flow. Nearly
    every station pair is highly correlated in raw flow at lag=0 simply
    because all stations follow the same daily rush-hour rhythm -- that
    shared seasonal component would otherwise swamp any genuine
    station-to-station propagation signal and the graph would degenerate
    into "everything is connected to everything at lag=0" (this is exactly
    what an earlier, non-deseasonalized version of this function produced
    empirically: ~88% of edges at lag=0 and none beyond lag=1). Working on
    residuals surfaces idiosyncratic surges that propagate between specific
    stations, which is the actual signal of interest here.

    This generalizes (and replaces) a same-timestep-only correlation graph:
    lag=0 is included as one candidate lag, but the graph can now also
    capture "a surge at station i tends to precede a surge at station j by
    ~k steps" -- e.g. a residential-area station feeding a business-district
    station with a commute-time offset -- which a lag-0-only correlation
    graph cannot represent, and which physical adjacency (direct rail
    connections) also cannot represent for stations that aren't adjacent but
    share a lagged ridership relationship.

    Returned as A[target, source] to match the attention convention used in
    MultiGraphAttention (row = receiver/query, column = source/key), i.e.
    A[j, i] = how much station j's forecast should attend to station i's past.

    Also returns best_lag[target, source] (in steps) for diagnostics/reporting
    -- e.g. plotting the empirical distribution of propagation lags found.
    """
    N = flow_train.shape[1]
    combined = flow_train.sum(axis=-1)  # (T, N) total station activity (in+out)

    seasonal_mean = np.zeros((n_slots, N), dtype=np.float32)
    for s in range(n_slots):
        sel = slot_id_train == s
        if sel.sum() > 0:
            seasonal_mean[s] = combined[sel].mean(axis=0)
    residual = combined - seasonal_mean[slot_id_train]  # (T, N), each station's own daily rhythm removed

    best_corr = np.zeros((N, N), dtype=np.float32)   # best_corr[target=j, source=i]
    best_lag = np.zeros((N, N), dtype=np.int32)

    for lag in range(0, max_lag + 1):
        if lag == 0:
            src, tgt = residual, residual
        else:
            src, tgt = residual[:-lag], residual[lag:]   # src = earlier window, tgt = later window
        if len(src) < 20:
            continue

        src_z = (src - src.mean(axis=0)) / (src.std(axis=0) + 1e-8)
        tgt_z = (tgt - tgt.mean(axis=0)) / (tgt.std(axis=0) + 1e-8)
        # corr[i, j] = corr(source station i's earlier values, target station j's later values)
        corr = (src_z.T @ tgt_z) / src_z.shape[0]
        corr = np.nan_to_num(corr, nan=0.0)
        if lag == 0:
            np.fill_diagonal(corr, 0.0)  # don't let self-correlation dominate its own row

        corr_as_target_source = corr.T  # [target=j, source=i] = corr(source=i, target=j)
        update = corr_as_target_source > best_corr
        best_corr = np.where(update, corr_as_target_source, best_corr)
        best_lag = np.where(update, lag, best_lag)

    A = np.zeros((N, N), dtype=np.float32)
    for j in range(N):
        top = np.argsort(-best_corr[j])[:top_k]
        A[j, top] = np.clip(best_corr[j, top], 0.0, None)

    return A.astype(np.float32), best_lag


def load_city(metro_data_dir: str, city: str) -> CityTensors:
    base = Path(metro_data_dir) / CITY_FILES[city]
    dyna = pd.read_csv(base / f"{CITY_FILES[city]}.dyna")
    rel = pd.read_csv(base / f"{CITY_FILES[city]}.rel")

    dyna["time"] = pd.to_datetime(dyna["time"])
    n_stations = dyna["entity_id"].nunique()

    inflow_col = "inflow" if "inflow" in dyna.columns else "in_flow"
    outflow_col = "outflow" if "outflow" in dyna.columns else "out_flow"

    inflow = dyna.pivot(index="time", columns="entity_id", values=inflow_col).sort_index().fillna(0)
    outflow = dyna.pivot(index="time", columns="entity_id", values=outflow_col).sort_index().fillna(0)
    time_index = inflow.index.tz_convert(None) if inflow.index.tz is not None else inflow.index

    flow = np.stack([inflow.values, outflow.values], axis=-1).astype(np.float32)  # (T, N, 2)

    minutes = time_index.hour * 60 + time_index.minute
    slot_id = ((minutes - minutes.min()) // 15).values
    n_slots = int(slot_id.max()) + 1
    daily_lag = n_slots  # one operating day, in steps

    adjacency_physical = _build_physical_adjacency(rel, n_stations)
    adjacency_similarity_precomputed = _build_precomputed_edge_adjacency(rel, n_stations, "similarity")
    adjacency_correlation_precomputed = _build_precomputed_edge_adjacency(rel, n_stations, "correlation")

    split = DATE_SPLITS.get(city)
    idx = {}
    if split is not None:
        train_mask = _date_mask(time_index, *split["train"])
        test_mask = _date_mask(time_index, *split["test"])
        official_val_mask = _date_mask(time_index, *split["official_val"])
        internal_fit_mask = _date_mask(time_index, *split["internal_fit"])
        internal_val_mask = _date_mask(time_index, *split["internal_val"])
        idx = {
            "train": np.where(train_mask)[0],
            "test": np.where(test_mask)[0],
            "official_val": np.where(official_val_mask)[0],
            "internal_fit": np.where(internal_fit_mask)[0],
            "internal_val": np.where(internal_val_mask)[0],
        }
    else:
        n = len(flow)
        train_end, val_end = int(n * 0.70), int(n * 0.85)
        all_idx = np.arange(n)
        idx = {
            "train": all_idx[:val_end],           # train+val combined for final refit
            "test": all_idx[val_end:],
            "official_val": all_idx[train_end:val_end],
            "internal_fit": all_idx[: int(train_end * 0.8)],
            "internal_val": all_idx[int(train_end * 0.8): train_end],
        }

    adjacency_corr, lag_matrix = _build_lagged_correlation_adjacency(
        flow[idx["train"]], slot_id[idx["train"]], n_slots
    )

    return CityTensors(
        city=city,
        flow=flow,
        time_index=time_index,
        adjacency_physical=torch.tensor(adjacency_physical, dtype=torch.float32),
        adjacency_corr=torch.tensor(adjacency_corr, dtype=torch.float32),
        lag_matrix=lag_matrix,
        num_stations=n_stations,
        slot_id=slot_id,
        n_slots=n_slots,
        daily_lag=daily_lag,
        idx=idx,
        adjacency_similarity_precomputed=(
            torch.tensor(adjacency_similarity_precomputed, dtype=torch.float32)
            if adjacency_similarity_precomputed is not None else None
        ),
        adjacency_correlation_precomputed=(
            torch.tensor(adjacency_correlation_precomputed, dtype=torch.float32)
            if adjacency_correlation_precomputed is not None else None
        ),
    )


def build_seasonal_profile(data: CityTensors, mask_idx: np.ndarray) -> np.ndarray:
    """Per-slot (time-of-day) station-level mean flow, fit only on the given indices."""
    profile = np.zeros((data.n_slots, data.num_stations, 2), dtype=np.float32)
    mask_set = np.zeros(len(data.flow), dtype=bool)
    mask_set[mask_idx] = True
    for s in range(data.n_slots):
        sel = np.where((data.slot_id == s) & mask_set)[0]
        if len(sel) > 0:
            profile[s] = data.flow[sel].mean(axis=0)
    return profile


def time_features(time_index: pd.DatetimeIndex) -> np.ndarray:
    m = time_index.hour * 60 + time_index.minute
    tod = 2 * math.pi * m.values / (24 * 60)
    dow = 2 * math.pi * time_index.dayofweek.values / 7
    return np.stack([np.sin(tod), np.cos(tod), np.sin(dow), np.cos(dow)], axis=-1).astype(np.float32)


def build_windows(
    data: CityTensors,
    idx_range: np.ndarray,
    seasonal_profile: np.ndarray,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
):
    """Builds (X, Y, time_feat, extra_feat, seasonal_horizon) sliding windows
    for the given index range. X/Y are raw counts (the model itself
    standardizes internally for the backbone while training the
    distributional head on raw counts, see model.py).

    extra_feat per window (aligned to the *last* lookback step, i.e. "now"):
      - seasonal mean at this slot (N, 2)
      - daily-lag flow, i.e. same time yesterday (N, 2), NaN-safe (zeros if unavailable)

    seasonal_horizon per window: (horizon, N, 2) -- the seasonal (historical
    time-of-day) mean flow at EACH future step being predicted (t, t+1, ...,
    t+horizon-1), not just at "now". This is what the model uses as a
    residual base: predicting a correction relative to this known, reliable
    per-step baseline is a much easier learning problem than predicting the
    raw value from scratch, especially with limited training data (see
    model.py's `use_seasonal_residual` for how this is used).
    """
    T = data.flow.shape[0]
    lo, hi = idx_range.min(), idx_range.max()
    ctx_lo = max(0, lo - lookback)
    t_range = list(range(max(lo, lookback), min(hi, T - horizon) + 1))

    tfeat_all = time_features(data.time_index)

    X, Y, TF, EXTRA, SEAS_HORIZON = [], [], [], [], []
    for t in t_range:
        X.append(data.flow[t - lookback:t])                 # (lookback, N, 2)
        Y.append(data.flow[t:t + horizon])                    # (horizon, N, 2)
        TF.append(tfeat_all[t - lookback:t])                   # (lookback, 4)
        seas = seasonal_profile[data.slot_id[t]]                # (N, 2)
        daily = data.flow[t - data.daily_lag] if t - data.daily_lag >= 0 else np.zeros((data.num_stations, 2), dtype=np.float32)
        EXTRA.append(np.concatenate([seas, daily], axis=-1))    # (N, 4)
        seas_h = np.stack([seasonal_profile[data.slot_id[t + h]] for h in range(horizon)], axis=0)  # (horizon, N, 2)
        SEAS_HORIZON.append(seas_h)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32),
        np.asarray(TF, dtype=np.float32),
        np.asarray(EXTRA, dtype=np.float32),
        np.asarray(SEAS_HORIZON, dtype=np.float32),
    )
