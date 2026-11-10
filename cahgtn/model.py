"""
cahgtn/model.py
================================================================
CA-HGTN: Count-Aware Heteroscedastic Graph-Temporal Network.

Backbone (multi-graph spatial attention + Transformer/GRU temporal
fusion) is a competent but not-particularly-novel spatiotemporal
encoder -- comparable in spirit to Graph WaveNet / PVCGN-style
models. The methodological contribution is the OUTPUT HEAD and LOSS:
instead of predicting a single point forecast trained with
MSE/MAE/Huber, the model predicts the parameters of a per-station,
per-horizon Negative Binomial distribution (mean mu, dispersion r)
and is trained with the NB negative log-likelihood. This directly
targets the heteroscedastic, count-valued nature of ridership data
(variance grows with the mean; small counts are proportionally much
noisier), which point-regression losses ignore.

Ablation switches are built in (see `head_type` and `use_multigraph`)
so a paper can report:
  - CA-HGTN (full)
  - CA-HGTN, point-regression head instead of NB (isolates the head's effect)
  - CA-HGTN, single-graph instead of multi-graph (isolates graph fusion's effect)
  - GRU-only baseline (no spatial structure at all, a sanity floor)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


GRAPH_MODES = ("full", "physical_only", "no_physical", "adaptive_only", "correlation_only")


# ----------------------------------------------------------------
# Spatial: multi-graph attention with learned per-layer gating
# ----------------------------------------------------------------
class MultiGraphAttention(nn.Module):
    """`graph_mode` controls which of the three graph views (physical
    topology, lag-correlation, learned adaptive) are fused:
      - "full": all three (the original use_multigraph=True) -- default.
      - "physical_only": physical topology only (the original
        use_multigraph=False / "single-graph" ablation).
      - "no_physical": correlation + adaptive only, NO physical topology.
        Added after the 5-seed Hangzhou run showed "physical_only"
        (MAE 30.86+-1.31) losing to a bare GRU with NO spatial structure at
        all (MAE 28.38+-0.13) -- i.e. physical adjacency alone was actively
        HURTING relative to no graph, while "full" (physical+correlation+
        adaptive together) clearly helped (27.34+-0.71). This mode tests
        whether physical topology is dead weight even inside the fused
        graph, or whether it's specifically harmful only in isolation.
      - "adaptive_only" / "correlation_only": isolate each remaining view
        individually, for a complete per-graph ablation.
    `use_multigraph` is kept as a deprecated alias: True maps to "full",
    False maps to "physical_only", so old code/configs keep working
    unchanged and existing saved results remain comparable."""

    def __init__(self, hidden_dim: int, num_stations: int, dropout: float = 0.3,
                 use_multigraph: bool | None = None, graph_mode: str | None = None,
                 edge_dropout: float = 0.0, correlation_temperature: float = 1.0):
        super().__init__()
        if graph_mode is None:
            graph_mode = "full" if (use_multigraph is None or use_multigraph) else "physical_only"
        assert graph_mode in GRAPH_MODES, f"graph_mode must be one of {GRAPH_MODES}, got {graph_mode!r}"
        self.graph_mode = graph_mode
        self.edge_dropout = edge_dropout
        self.correlation_temperature = correlation_temperature

        self.node_emb1 = nn.Parameter(torch.randn(num_stations, 16) * 0.01)
        self.node_emb2 = nn.Parameter(torch.randn(num_stations, 16) * 0.01)

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

        n_graphs = 3 if graph_mode == "full" else 1
        self.graph_gate = nn.Parameter(torch.ones(n_graphs))

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # Caches for the row-normalized physical/correlation graphs: unlike
        # the adaptive graph, these don't depend on any learned parameter --
        # adj_physical and adj_corr are fixed inputs for an entire training
        # run, so _row_normalize(adj_physical) and _row_normalize(adj_corr)
        # were being recomputed identically on EVERY forward call (every
        # batch, every epoch) for no reason. Keyed on id() (object identity,
        # not value equality) rather than caching unconditionally: if a
        # genuinely different adjacency tensor is ever passed to the same
        # instance (this project now has variants that swap in different
        # precomputed graphs), a false cache HIT would silently serve a
        # stale, wrong normalization -- id()-keying means a different
        # tensor always correctly invalidates and recomputes; the only
        # thing that gets skipped is recomputing for the SAME object,
        # which is safe by construction.
        self._physical_norm_cache = None
        self._physical_norm_cache_id = None
        self._corr_norm_cache = None
        self._corr_norm_cache_id = None

    def _cached_row_normalize_physical(self, adj_physical: torch.Tensor) -> torch.Tensor:
        if self._physical_norm_cache_id != id(adj_physical):
            self._physical_norm_cache = self._row_normalize(adj_physical)
            self._physical_norm_cache_id = id(adj_physical)
        return self._physical_norm_cache

    def _apply_edge_dropout(self, adj: torch.Tensor) -> torch.Tensor:
        """Randomly zeros a fraction (self.edge_dropout probability each) of
        entries in an already-normalized graph, freshly random every forward
        call, TRAINING ONLY (a no-op in eval mode or when edge_dropout=0,
        matching standard nn.Dropout semantics). Adapted from DropEdge
        (Rong et al., ICLR 2020, "DropEdge: Towards Deep Graph Convolutional
        Networks on Node Classification") -- used here as much for its
        graph-specific data-augmentation framing (forcing the model not to
        rely on any single edge being present every step) as for DropEdge's
        originally-studied over-smoothing benefit. Directly motivated by
        this project's own diagnosed instability: `no-physical` showed
        cross-seed variance nearly as bad as `point head` (+-4.81 vs
        +-5.55), hypothesized to trace to the adaptive graph's freely-
        learned embeddings converging to inconsistent structures across
        seeds -- this is a complementary way to address that same
        diagnosed problem, alongside the entropy regularization already
        built for it.

        No renormalization afterward: `fused_mask` enters attention as an
        ADDITIVE log-bias (see forward()'s `torch.log(fused_mask + 1e-6)`),
        and the downstream softmax renormalizes regardless of what row-sum
        it's fed -- so dropping entries here doesn't leave the attention
        distribution improperly scaled the way it would in a architecture
        that used the graph as a direct multiplicative mask instead.

        Composes with the row-normalization caching above by design: the
        CACHED value is the deterministic base normalization (unchanged by
        this method), and this dropout mask is drawn fresh on top of it
        every call -- so training still gets genuine, per-step randomness
        while the expensive part (row-sum-based normalization) still only
        happens once per unique input tensor, not once per call."""
        if not self.training or self.edge_dropout <= 0.0:
            return adj
        keep_mask = (torch.rand_like(adj) >= self.edge_dropout).float()
        return adj * keep_mask

    def _cached_row_normalize_corr(self, adj_corr: torch.Tensor) -> torch.Tensor:
        if self._corr_norm_cache_id != id(adj_corr):
            self._corr_norm_cache = self._row_normalize(adj_corr, temperature=self.correlation_temperature)
            self._corr_norm_cache_id = id(adj_corr)
        return self._corr_norm_cache

    def _adaptive_adj(self) -> torch.Tensor:
        logits = F.relu(self.node_emb1 @ self.node_emb2.t())
        return F.softmax(logits, dim=-1)

    def adaptive_entropy(self) -> torch.Tensor | None:
        """Returns the mean row-wise entropy of the adaptive graph, or None
        if this layer doesn't use one. Used as an optional regularization
        signal -- see CAHGTN.adaptive_graph_entropy_regularization for the
        motivation and how it's used."""
        if self.graph_mode not in ("full", "no_physical", "adaptive_only"):
            return None
        adaptive_adj = self._adaptive_adj()
        return -(adaptive_adj * torch.log(adaptive_adj + 1e-9)).sum(dim=-1).mean()

    @staticmethod
    def _row_normalize(adj: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """temperature=1.0 (default): EXACTLY the original behavior, unchanged
        code path -- byte-identical by construction, not just empirically.
        temperature>1.0: raises entries to the 1/temperature power before
        normalizing, compressing the gap between a row's dominant entry and
        the rest (e.g. temperature=2.0 takes a square root first). Directly
        targets the station-25 mechanism: a row-normalized correlation
        weight becomes a direct additive log-bias to attention (see
        forward()) -- linear row-normalization doesn't soften a dominant
        entry's share, so a single strong correlation neighbor can force a
        large attention boost regardless of what Q/K similarity says.
        Clamp only applied when temperature != 1.0, so the default path
        never touches values that weren't touched before."""
        adj = adj + torch.eye(adj.shape[-1], device=adj.device)
        if temperature != 1.0:
            adj = adj.clamp(min=1e-8) ** (1.0 / temperature)
        row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return adj / row_sum

    def forward(self, h: torch.Tensor, adj_physical: torch.Tensor, adj_corr: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        Q, K, V = self.q(h), self.k(h), self.v(h)
        attn_logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(D)

        if self.graph_mode == "full":
            gates = F.softmax(self.graph_gate, dim=0)
            fused_mask = (
                gates[0] * self._apply_edge_dropout(self._cached_row_normalize_physical(adj_physical))
                + gates[1] * self._apply_edge_dropout(self._cached_row_normalize_corr(adj_corr))
                + gates[2] * self._apply_edge_dropout(self._adaptive_adj())
            )
        elif self.graph_mode == "physical_only":
            fused_mask = self._apply_edge_dropout(self._cached_row_normalize_physical(adj_physical))
        elif self.graph_mode == "correlation_only":
            fused_mask = self._apply_edge_dropout(self._cached_row_normalize_corr(adj_corr))
        elif self.graph_mode == "adaptive_only":
            fused_mask = self._apply_edge_dropout(self._adaptive_adj())
        elif self.graph_mode == "no_physical":
            gates = F.softmax(self.graph_gate, dim=0)  # 1 gate value here, softmax is a no-op but keeps the param used
            fused_mask = (self._apply_edge_dropout(self._cached_row_normalize_corr(adj_corr))
                          + self._apply_edge_dropout(self._adaptive_adj()))
        else:
            raise ValueError(f"unknown graph_mode {self.graph_mode!r}")

        fused_mask = fused_mask.unsqueeze(0)
        attn = F.softmax(attn_logits + torch.log(fused_mask + 1e-6), dim=-1)
        attn = self.dropout(attn)
        out = self.out_proj(torch.matmul(attn, V))
        return self.norm(h + out)


# ----------------------------------------------------------------
# Temporal: Transformer branch + GRU branch, fused with a learned gate
# ----------------------------------------------------------------
class TemporalFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trans_out = self.transformer(x)
        gru_out, _ = self.gru(x)
        gru_out = self.gru_proj(gru_out)
        gate = torch.sigmoid(self.gate(torch.cat([trans_out, gru_out], dim=-1)))
        fused = gate * trans_out + (1 - gate) * gru_out
        weights = F.softmax(fused.mean(dim=-1), dim=-1).unsqueeze(-1)
        return (fused * weights).sum(dim=1)


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, in_dim: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.proj(feat)


def _zero_init_head(layer: nn.Linear):
    """Zero-inits a Linear layer's weight AND bias, so its output is EXACTLY
    0 for any input at initialization. Used for correction/delta heads under
    the seasonal-residual mechanism -- see the call sites in CAHGTN and
    GRUOnlyBaseline for the full rationale."""
    with torch.no_grad():
        layer.weight.data[...] = 0.0
        layer.bias.data[...] = 0.0


def _init_softplus_output_bias(layer: nn.Linear, target: tuple | None, horizon: int):
    """Initializes a Linear layer's bias so that softplus(bias) starts near
    `target` = (mean_inflow, mean_outflow) instead of near softplus(0) = log(2)
    = 0.69, for every horizon step.

    Why this matters: real ridership flow averages in the hundreds, but a
    freshly initialized Linear layer (small random weights, zero bias)
    outputs values near 0, so softplus(~0) = 0.69 -- roughly 300x too small.
    The first gradient step then has to correct a huge scale error in one
    shot, which tends to produce a large, noisy update; if that noisy first
    epoch happens to look better than the next several (normal-sized) epochs
    purely by chance, early stopping can trigger almost immediately, before
    the model has had a real chance to learn anything -- a well-known
    failure mode for count-regression output heads. Initializing the bias to
    the right scale up front avoids needing that catastrophic first update.

    For large x, softplus(x) ~= x, so setting bias ~= target directly (rather
    than solving log(exp(target)-1) exactly) is numerically fine for
    realistic flow magnitudes.
    """
    if target is None:
        return
    mean_in, mean_out = target
    with torch.no_grad():
        bias = layer.bias.data
        for h in range(horizon):
            bias[h * 2 + 0] = mean_in
            bias[h * 2 + 1] = mean_out


# ----------------------------------------------------------------
# Full model
# ----------------------------------------------------------------
class CAHGTN(nn.Module):
    """
    x:               (B, T, N, 2) raw inflow/outflow counts (recent lookback window)
    time_feat:       (B, T, 4) sin/cos time-of-day & day-of-week, aligned to x
    extra:           (B, N, 4) [seasonal_mean_in, seasonal_mean_out, daily_lag_in, daily_lag_out] at "now"
    seasonal_horizon:(B, horizon, N, 2) seasonal (historical time-of-day) mean flow
                     at EACH future step being predicted -- see data_pipeline.build_windows.
    adj_*:           (N, N)

    head_type: "nb" (Negative Binomial distributional head, the novel
               contribution) or "point" (plain regression head, for ablation).
    graph_mode: which graph view(s) the spatial attention uses -- one of
        MultiGraphAttention.GRAPH_MODES ("full", "physical_only",
        "no_physical", "adaptive_only", "correlation_only"). `use_multigraph`
        is kept as a deprecated alias (True="full", False="physical_only")
        so existing configs/results stay valid; pass `graph_mode` directly
        for the newer ablation modes.
    use_seasonal_residual: toggles the residual-around-seasonal-baseline
        mechanism (see forward() docstring below) vs. predicting the absolute
        value from scratch, for ablation.

    Returns a dict of horizon-stacked tensors:
      head_type="nb":    {"mu": (B,H,N,2), "r": (B,H,N,2)}
      head_type="point": {"pred": (B,H,N,2)}
    """

    RESIDUAL_FLOOR = 1.0     # avoids multiplying by exactly 0 for genuinely-zero-traffic (station, slot) cells
    CORRECTION_CLAMP = 3.0   # bounds the multiplicative correction to roughly [exp(-3), exp(3)] = [0.05x, 20x]

    def __init__(
        self,
        num_stations: int,
        hidden_dim: int = 64,
        horizon: int = 4,
        spatial_layers: int = 2,
        dropout: float = 0.3,
        head_type: str = "nb",
        use_multigraph: bool | None = None,
        graph_mode: str | None = None,
        use_seasonal_residual: bool = True,
        output_bias_init: tuple | None = None,
        edge_dropout: float = 0.0,
        correlation_temperature: float = 1.0,
    ):
        super().__init__()
        assert head_type in ("nb", "point")
        self.horizon = horizon
        self.head_type = head_type
        self.num_stations = num_stations
        self.use_seasonal_residual = use_seasonal_residual

        # Log1p-compress raw counts before the shared encoder for numerical
        # stability (the distributional head still operates on/targets raw
        # counts via mu/r, this is purely an encoder-input transform).
        self.input_proj = nn.Linear(2, hidden_dim)
        self.time_emb = TimeEmbedding(hidden_dim, in_dim=4)
        self.extra_proj = nn.Linear(4, hidden_dim)

        self.spatial_layers = nn.ModuleList([
            MultiGraphAttention(hidden_dim, num_stations, dropout,
                                 use_multigraph=use_multigraph, graph_mode=graph_mode,
                                 edge_dropout=edge_dropout, correlation_temperature=correlation_temperature)
            for _ in range(spatial_layers)
        ])
        self.temporal = TemporalFusion(hidden_dim, dropout=dropout)

        self.trunk = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        if head_type == "nb":
            # mu (mean) and r (dispersion) per direction (in/out) per horizon step.
            self.mu_head = nn.Linear(hidden_dim, horizon * 2)
            self.r_head = nn.Linear(hidden_dim, horizon * 2)
            _init_softplus_output_bias(self.r_head, (10.0, 10.0), horizon)  # moderate starting dispersion
            if use_seasonal_residual:
                # Zero-init the correction head's weight AND bias -- not just
                # "small random" -- so correction is EXACTLY 0 for every
                # input at initialization, guaranteeing mu starts EXACTLY at
                # the seasonal baseline (not just approximately). This is
                # the standard "zero-init the last layer of a residual/delta
                # head" trick (also used e.g. in ResNets and RL policy
                # heads): safe here specifically because this is an OUTPUT
                # layer with a distinct target per output unit, so gradients
                # differentiate across units from the first backward pass --
                # the usual "dead symmetric units" risk of zero-init is a
                # hidden-layer problem, not an output-layer one.
                _zero_init_head(self.mu_head)
            else:
                # Only needed in the old absolute-value mode: without a
                # seasonal baseline to start from, the raw softplus output
                # otherwise starts near softplus(0)=0.69 against a target
                # averaging in the hundreds.
                _init_softplus_output_bias(self.mu_head, output_bias_init, horizon)
        else:
            self.point_head = nn.Linear(hidden_dim, horizon * 2)
            if use_seasonal_residual:
                _zero_init_head(self.point_head)
            else:
                _init_softplus_output_bias(self.point_head, output_bias_init, horizon)

    def adaptive_graph_entropy_regularization(self) -> torch.Tensor | None:
        """Auxiliary, OPT-IN regularization term (mean of -entropy across
        spatial layers that use an adaptive graph; None if none do).
        Minimizing this term pushes the adaptive graph's learned structure
        toward higher entropy (closer to uniform) unless the training
        signal strongly justifies committing to a peaked structure.

        Motivation: the 5-seed Hangzhou run showed `no-physical`
        (correlation+adaptive graphs, no physical topology) with MAE
        variance across seeds (+-4.81) nearly as large as `point head`
        (+-5.55), while `single-graph` (physical topology ALONE, no
        adaptive graph at all) was stable (+-1.31). Hypothesis: the
        adaptive graph's freely-learned node embeddings can converge to
        meaningfully different structures across random seeds, and that
        instability propagates into the final predictions -- especially
        without physical topology to anchor it.

        This is a hypothesis motivated by the observed pattern, not a
        verified fix -- it's wired in as strictly OPT-IN (default
        coefficient 0.0 in train.py, so existing behavior and every
        already-obtained result are completely unaffected unless you
        explicitly enable it) specifically so it can be tested cheaply
        (e.g. a quick no-physical smoke-test with vs. without it, checking
        whether cross-seed variance actually drops) before trusting it
        changes the real margin rather than just adding an unproven extra
        term to the loss.
        """
        entropies = [e for e in (layer.adaptive_entropy() for layer in self.spatial_layers) if e is not None]
        if not entropies:
            return None
        return -sum(entropies) / len(entropies)

    def forward(self, x: torch.Tensor, time_feat: torch.Tensor, extra: torch.Tensor,
                seasonal_horizon: torch.Tensor,
                adj_physical: torch.Tensor, adj_corr: torch.Tensor):
        B, T, N, _ = x.shape

        h = self.input_proj(torch.log1p(x))
        h = h + self.time_emb(time_feat).unsqueeze(2)

        h_flat = h.reshape(B * T, N, -1)
        for layer in self.spatial_layers:
            h_flat = layer(h_flat, adj_physical, adj_corr)
        h = h_flat.reshape(B, T, N, -1)

        h_seq = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        pooled = self.temporal(h_seq)                          # (B*N, D)
        pooled = pooled.reshape(B, N, -1)
        pooled = pooled + self.extra_proj(torch.log1p(extra))    # inject seasonal/daily-lag context
        pooled = self.trunk(pooled)                              # (B, N, D)

        if self.head_type == "nb":
            raw_mu = self.mu_head(pooled)                          # (B, N, H*2), unconstrained
            r = F.softplus(self.r_head(pooled)) + 1e-3
            raw_mu = raw_mu.reshape(B, N, self.horizon, 2).permute(0, 2, 1, 3)   # (B, H, N, 2)
            r = r.reshape(B, N, self.horizon, 2).permute(0, 2, 1, 3)

            if self.use_seasonal_residual:
                # mu = seasonal baseline * exp(bounded correction). At
                # initialization (correction ~= 0), mu ~= seasonal_horizon --
                # the model starts from a strong, reliable historical prior
                # instead of having to learn the whole scale of ridership
                # (0 to several thousand) from scratch. Training only needs
                # to learn the (much smaller, much more learnable) deviation.
                correction = torch.clamp(raw_mu, -self.CORRECTION_CLAMP, self.CORRECTION_CLAMP)
                mu = (seasonal_horizon + self.RESIDUAL_FLOOR) * torch.exp(correction)
            else:
                mu = F.softplus(raw_mu) + 1e-3
            return {"mu": mu, "r": r}
        else:
            raw_pred = self.point_head(pooled)
            raw_pred = raw_pred.reshape(B, N, self.horizon, 2).permute(0, 2, 1, 3)
            if self.use_seasonal_residual:
                correction = torch.clamp(raw_pred, -self.CORRECTION_CLAMP, self.CORRECTION_CLAMP)
                pred = (seasonal_horizon + self.RESIDUAL_FLOOR) * torch.exp(correction)
            else:
                pred = F.softplus(raw_pred)
            return {"pred": pred}


class GRUOnlyBaseline(nn.Module):
    """Sanity-floor ablation: no graph/attention structure at all, per-station
    GRU only. Uses the SAME seasonal-residual mechanism as CAHGTN (when
    enabled) so the ablation cleanly isolates "does the graph + Transformer
    add value on top of the residual trick", rather than conflating that
    question with "does the residual trick itself help"."""

    RESIDUAL_FLOOR = 1.0
    CORRECTION_CLAMP = 3.0

    def __init__(self, hidden_dim: int = 64, horizon: int = 4, output_bias_init: tuple | None = None,
                 use_seasonal_residual: bool = True):
        super().__init__()
        self.horizon = horizon
        self.use_seasonal_residual = use_seasonal_residual
        self.input_proj = nn.Linear(6, hidden_dim)  # flow(2) + time_feat(4)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, horizon * 2)
        if use_seasonal_residual:
            _zero_init_head(self.head)
        else:
            _init_softplus_output_bias(self.head, output_bias_init, horizon)

    def forward(self, x, time_feat, extra=None, seasonal_horizon=None, adj_physical=None, adj_corr=None):
        B, T, N, _ = x.shape
        tf = time_feat.unsqueeze(2).expand(-1, -1, N, -1)
        inp = torch.cat([torch.log1p(x), tf], dim=-1)          # (B,T,N,6)
        inp = self.input_proj(inp)                              # (B,T,N,hidden_dim)
        inp = inp.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        _, hN = self.gru(inp)
        raw_pred = self.head(hN[-1])
        raw_pred = raw_pred.reshape(B, N, self.horizon, 2).permute(0, 2, 1, 3)
        if self.use_seasonal_residual:
            correction = torch.clamp(raw_pred, -self.CORRECTION_CLAMP, self.CORRECTION_CLAMP)
            pred = (seasonal_horizon + self.RESIDUAL_FLOOR) * torch.exp(correction)
        else:
            pred = F.softplus(raw_pred)
        return {"pred": pred}
