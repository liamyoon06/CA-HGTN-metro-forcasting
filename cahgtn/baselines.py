"""
cahgtn/baselines.py
================================================================
Standard, widely-used baseline architecture, implemented under this
project's EXACT data pipeline, training loop, and evaluation protocol
(same date-anchored splits, same features, same metrics) -- built
specifically to resolve the cross-paper SOTA-number inconsistency found
in this project (STR-GODEs' cited PVCGN: MAE 32.68 vs. Parallel-MGCN's
cited PVCGN: MAE 22.20 -- a ~30% gap for the claimed same model on the
claimed same benchmark). Comparing against *reported numbers from other
papers* is demonstrably unreliable here; comparing against a baseline
reproduced under OUR OWN controlled protocol is not.

GCNGRUBaseline: one Kipf & Welling-style graph convolution (using the
PHYSICAL topology graph only, as is conventional for this architecture
family) applied independently at each lookback timestep, followed by a
GRU over the lookback window, and a linear output head. This is
deliberately much simpler than CA-HGTN's multi-graph attention +
Transformer/GRU backbone: the GCN+GRU pattern predates and is the direct
architectural ancestor of PVCGN, DCRNN, and STGCN, and is a standard
"simple but real" baseline choice in this literature.

IMPORTANT DESIGN DECISION: this baseline still uses the seasonal-residual
output mechanism (predicting a bounded correction around each station's
historical time-of-day baseline, rather than the absolute value from
scratch) and the same zero-init trick as CA-HGTN. Without it, this
project's own earlier experiments showed EVERY architecture tried --
including CA-HGTN itself -- collapsing to a near-constant output on this
data, for reasons unrelated to architectural sophistication (a cold-start
initialization problem, not an architecture problem; see model.py's
CAHGTN docstring). Giving the baseline the SAME output mechanism isolates
the actual question this comparison is for: does CA-HGTN's extra
architectural complexity earn its keep over a standard, simple baseline,
holding the now-proven-necessary residual trick and identical data/
protocol constant? Comparing a from-scratch-initialized GCN+GRU against a
residual-initialized CA-HGTN would confound that question with "does the
residual trick matter," which we already know the answer to.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import _zero_init_head, _init_softplus_output_bias

TEMPORAL_MODES = ("plain", "bidirectional_attention", "stacked_bidir_uni_attention")


class GCNLayer(nn.Module):
    """Standard Kipf & Welling graph convolution: H' = ReLU(A_hat H W),
    where A_hat is the symmetrically-normalized adjacency with self-loops:
    A_hat = D^-1/2 (A + I) D^-1/2."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    @staticmethod
    def _normalize(adj: torch.Tensor) -> torch.Tensor:
        N = adj.shape[-1]
        adj = adj + torch.eye(N, device=adj.device)
        deg = adj.sum(dim=-1)
        deg_inv_sqrt = deg.clamp(min=1e-6) ** -0.5
        return adj * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(0)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """h: (B, N, D_in), adj: (N, N) -> (B, N, D_out)"""
        a_hat = self._normalize(adj)
        agg = torch.matmul(a_hat.unsqueeze(0), h)  # (B, N, D_in)
        return F.relu(self.linear(agg))


class GCNGRUBaseline(nn.Module):
    """Same forward() signature as CAHGTN/GRUOnlyBaseline (accepts and
    ignores adj_corr, since this baseline is single-graph by design) so it
    plugs directly into the existing train.py / predict_all machinery with
    no changes needed there."""

    RESIDUAL_FLOOR = 1.0
    CORRECTION_CLAMP = 3.0

    def __init__(self, num_stations: int, hidden_dim: int = 64, horizon: int = 4,
                 use_seasonal_residual: bool = True, output_bias_init: tuple | None = None,
                 bidirectional_attention: bool = False, temporal_mode: str | None = None):
        super().__init__()
        self.horizon = horizon
        self.use_seasonal_residual = use_seasonal_residual

        # `temporal_mode` supersedes `bidirectional_attention`, kept as a
        # backward-compatible alias -- mirrors model.py's own graph_mode/
        # use_multigraph pattern for the same reason: `bidirectional_attention`
        # was a plain bool before a second, richer variant existed, and
        # letting boolean flags multiply per new idea doesn't compose. Every
        # existing call site (bidirectional_attention=True or =False,
        # explicit or default) maps to the exact same temporal_mode it always
        # implied, so nothing about prior results or notebook cells changes.
        if temporal_mode is None:
            temporal_mode = "bidirectional_attention" if bidirectional_attention else "plain"
        assert temporal_mode in TEMPORAL_MODES, f"temporal_mode must be one of {TEMPORAL_MODES}, got {temporal_mode!r}"
        self.temporal_mode = temporal_mode

        self.gcn = GCNLayer(2, hidden_dim)          # per-timestep spatial conv on raw (log1p) flow
        self.time_proj = nn.Linear(4, hidden_dim)     # time-of-day/day-of-week features

        # Three temporal components, same hidden_dim/GCN/graph across all of
        # them -- isolates the temporal-component-sophistication hypothesis
        # cleanly, one step at a time:
        #   "plain": original unidirectional GRU + last-hidden-state.
        #   "bidirectional_attention": one bidirectional GRU layer + learned
        #     attention pooling. REAL RESULT (5 seeds, Hangzhou): 27.98+-0.20
        #     MAE vs "plain"'s 28.54+-0.34 -- for ~2x parameters, a bigger,
        #     more parameter-efficient improvement than the separate
        #     capacity-scaling check bought for ~15x parameters (0.37 MAE at
        #     hidden_dim=256, though that specific comparison was only 1
        #     seed there, not 5 -- see PAPER_PLAN.md). Still small relative
        #     to the ~6-point gap to PB-GRU -- evidence for one contributing
        #     factor, not most of the explanation.
        #   "stacked_bidir_uni_attention": a bidirectional GRU layer followed
        #     by a unidirectional GRU layer over its output, then attention
        #     pooling -- a genuine, reasonable interpretation of PB-GRU's
        #     own description of its temporal component ("stacked
        #     bidirectional-unidirectional GRU"), built from their paper's
        #     text, not their source code (not available) -- stated plainly
        #     as an interpretation, not a guaranteed exact replication.
        #     Tests whether stacking captures more of the same effect that a
        #     single bidirectional layer already showed, as its own
        #     hypothesis worth checking on its own terms, not toward a
        #     specific target number.
        if temporal_mode == "stacked_bidir_uni_attention":
            self.gru_bidir = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.gru_uni = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
            self.gru = None
            self.attn_pool = nn.Linear(hidden_dim, 1)
            head_in_dim = hidden_dim
        elif temporal_mode == "bidirectional_attention":
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.gru_bidir = self.gru_uni = None
            self.attn_pool = nn.Linear(hidden_dim * 2, 1)
            head_in_dim = hidden_dim * 2
        else:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.gru_bidir = self.gru_uni = None
            self.attn_pool = None
            head_in_dim = hidden_dim
        self.head = nn.Linear(head_in_dim, horizon * 2)

        if use_seasonal_residual:
            _zero_init_head(self.head)
        else:
            _init_softplus_output_bias(self.head, output_bias_init, horizon)

    def forward(self, x, time_feat, extra=None, seasonal_horizon=None,
                adj_physical=None, adj_corr=None):
        """x: (B,T,N,2), time_feat: (B,T,4), seasonal_horizon: (B,H,N,2),
        adj_physical: (N,N) -- standard GCN+GRU uses ONLY the physical
        graph, as is conventional for this architecture family (multi-graph
        fusion is CA-HGTN's contribution, not this baseline's); adj_corr is
        accepted and ignored purely for call-signature compatibility."""
        B, T, N, _ = x.shape
        h_log = torch.log1p(x)  # (B,T,N,2)

        # Vectorized: fold T into the batch dimension for ONE gcn() call,
        # instead of T separate Python-loop calls. Mathematically identical
        # to the per-timestep loop this replaced -- GCNLayer applies the
        # SAME graph convolution independently to each entry along its
        # leading dimension regardless of whether that dimension represents
        # "batch" or "batch*time", so folding T into B changes nothing
        # about the computation, only how many Python-level calls it takes
        # (1 instead of T) -- verified byte-identical against the original
        # loop version before this replaced it.
        h_flat = h_log.reshape(B * T, N, 2)
        h_flat = self.gcn(h_flat, adj_physical)          # (B*T,N,hidden)
        h = h_flat.reshape(B, T, N, -1)                    # (B,T,N,hidden)

        tf = self.time_proj(time_feat).unsqueeze(2)  # (B,T,1,hidden)
        h = h + tf

        h_seq = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)  # (B*N,T,hidden)

        if self.temporal_mode == "stacked_bidir_uni_attention":
            bidir_out, _ = self.gru_bidir(h_seq)         # (B*N, T, hidden*2)
            uni_out, _ = self.gru_uni(bidir_out)           # (B*N, T, hidden)
            scores = self.attn_pool(uni_out).squeeze(-1)    # (B*N, T)
            weights = F.softmax(scores, dim=-1)
            pooled = (uni_out * weights.unsqueeze(-1)).sum(dim=1)  # (B*N, hidden)
        elif self.temporal_mode == "bidirectional_attention":
            gru_out, _ = self.gru(h_seq)              # (B*N, T, hidden*2)
            scores = self.attn_pool(gru_out).squeeze(-1)   # (B*N, T)
            weights = F.softmax(scores, dim=-1)
            pooled = (gru_out * weights.unsqueeze(-1)).sum(dim=1)  # (B*N, hidden*2)
        else:
            _, hN = self.gru(h_seq)
            pooled = hN[-1]  # (B*N, hidden) -- IDENTICAL to prior behavior

        raw = self.head(pooled)  # (B*N, horizon*2)
        raw = raw.reshape(B, N, self.horizon, 2).permute(0, 2, 1, 3)  # (B,H,N,2)

        if self.use_seasonal_residual:
            correction = torch.clamp(raw, -self.CORRECTION_CLAMP, self.CORRECTION_CLAMP)
            pred = (seasonal_horizon + self.RESIDUAL_FLOOR) * torch.exp(correction)
        else:
            pred = F.softplus(raw)
        return {"pred": pred}
