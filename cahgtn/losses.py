"""
cahgtn/losses.py
================================================================
Negative Binomial negative log-likelihood, parameterized by
mean (mu) and dispersion (r), where variance = mu + mu^2 / r.
As r -> infinity this reduces to Poisson (variance = mu); smaller
r captures extra overdispersion often seen in real ridership counts
(bursty arrivals, event effects) beyond what a pure Poisson assumes.

This is the central novel training objective: instead of directly
penalizing (pred - true)^2 or |pred - true| uniformly across all
count magnitudes, the NB-NLL naturally scales the effective penalty
by the predicted variance at that magnitude -- i.e. the loss itself
is heteroscedasticity-aware, which is exactly what plain point
regression on count data is missing.

IMPORTANT ADDITION after diagnosing collapsed/compressed predictions
on real HZMetro data (max flow ~3800, median ~140 -- a heavy right
tail): two related failure modes showed up and are addressed below.

1. NB-NLL alone lets the model "cheat": for a hard-to-predict spike,
   it can predict a low mu and just inflate r (dispersion) instead of
   actually learning to predict the spike's magnitude -- that keeps
   likelihood high without ever getting the mean right. `combined_nb_loss`
   adds a direct, weighted penalty on |mu - target| that this escape
   hatch doesn't help with.
2. Unweighted Huber/MAE on a heavy-tailed target is dominated by the
   much more numerous near-median examples, so "always predict near
   the median" is the loss-minimizing strategy even though it's
   catastrophic on the rare large spikes that matter most for
   MAE/RMSE at evaluation time. `weighted_point_loss` upweights
   larger true values so the optimizer can't get away with ignoring
   them.
"""

from __future__ import annotations

import torch


def negative_binomial_nll(mu: torch.Tensor, r: torch.Tensor, target: torch.Tensor,
                           beta: float = 0.0, r_reference: float = 10.0) -> torch.Tensor:
    """mu, r, target: same shape, all >= 0 (target should be raw counts).

    `beta` (default 0.0, an EXACT no-op reproducing prior behavior): applies
    a beta-NLL-style reweighting, adapting Seitzer, Tavakoli, Antic & Martius
    (ICLR 2022, "On the Pitfalls of Heteroscedastic Uncertainty Estimation
    with Probabilistic Neural Networks") from their Gaussian heteroscedastic
    case to this Negative-Binomial one. They formally diagnosed a pathology
    in plain heteroscedastic NLL training: the mean's gradient is naturally
    suppressed wherever the model's own dispersion estimate says "this is
    uncertain," and since that suppression makes it CHEAPER to leave the
    mean wrong (rather than fix it), the dynamic self-reinforces. This is
    exactly the "escape hatch" this project diagnosed independently, early
    on: verified numerically that shrinking r from 10 to 0.5, for a
    badly-wrong mu=254 against true=3809, cuts the loss ~7.7x without ever
    correcting mu (see PAPER_PLAN.md). Their fix multiplies each example's
    NLL by its own dispersion, DETACHED, raised to `beta`; beta=0 recovers
    plain NLL, larger beta increasingly counteracts the suppression. Applying
    this to our NB parameterization is a genuine adaptation, not a
    copy-paste -- NB's gradient structure differs from Gaussian's -- and was
    verified numerically for our specific case, not assumed by analogy: the
    same escape-hatch scenario above goes from ~7.7x cheaper (beta=0) to
    fully neutralized around beta=0.75 to ~50x MORE expensive (beta=2.0),
    a clean monotonic transition. `r.detach()` mirrors their exact
    safeguard -- without it, the reweighting factor would itself receive
    gradient feedback through r, creating a new, second-order coupling
    between the loss's own shape and the parameter it's trying to keep
    honest.

    `r_reference` (default 10.0, matching this project's own r_head
    initialization convention -- see model.py's _init_softplus_output_bias):
    the reweighting is normalized against this FIXED value, not the current
    batch's own mean r. An earlier version of this normalized against the
    batch's own mean instead, which has a real bug: a batch with little
    internal variation in r (caught directly via testing, not by
    inspection) silently neutralizes the reweighting entirely, since
    dividing a near-uniform tensor by its own near-uniform mean collapses
    to ~1.0 everywhere regardless of beta. Normalizing against a fixed
    reference instead means reweight=1.0 exactly when r==r_reference
    (matching beta=0's implicit baseline at that point), independent of
    whatever else is in the same batch.

    This project's own literature search found no prior use of beta-NLL
    (or an NB adaptation of it) in transit ridership forecasting, which
    has no distributional heads to begin with.
    """
    eps = 1e-8
    log_r = torch.log(r + eps)
    log_mu = torch.log(mu + eps)
    log_r_mu = torch.log(r + mu + eps)

    ll_per_cell = (
        torch.lgamma(target + r)
        - torch.lgamma(r)
        - torch.lgamma(target + 1)
        + r * (log_r - log_r_mu)
        + target * (log_mu - log_r_mu)
    )
    nll_per_cell = -ll_per_cell

    if beta > 0:
        reweight = ((r.detach() + eps) / r_reference) ** (-beta)
        nll_per_cell = nll_per_cell * reweight

    return nll_per_cell.mean()


def masked_huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return torch.nn.functional.huber_loss(pred, target, delta=delta)


def weighted_point_loss(pred: torch.Tensor, target: torch.Tensor, weight_power: float = 0.5,
                         delta: float = 1.0) -> torch.Tensor:
    """Huber loss with each example weighted by target**weight_power, so
    large-flow examples (the rare spikes a heavy-tailed distribution has a
    lot of, relatively speaking, in terms of their contribution to MAE/RMSE)
    aren't drowned out by the much more numerous near-median examples.
    weight_power=0.5 (sqrt) is a moderate choice: full linear weighting
    (weight_power=1) would let the rare largest spikes dominate the loss
    almost entirely, which risks the opposite failure (ignoring the bulk of
    ordinary examples to chase outliers)."""
    weight = torch.clamp(target, min=1.0) ** weight_power
    weight = weight / weight.mean()  # keep overall loss magnitude comparable across configs

    diff = pred - target
    abs_diff = torch.abs(diff)
    quad = torch.clamp(abs_diff, max=delta)
    lin = abs_diff - quad
    per_example = 0.5 * quad ** 2 + delta * lin
    return (per_example * weight).mean()


def combined_nb_loss(mu: torch.Tensor, r: torch.Tensor, target: torch.Tensor,
                      aux_weight: float = 0.15, weight_power: float = 0.5,
                      beta: float = 0.0, r_reference: float = 10.0) -> torch.Tensor:
    """NB-NLL plus a weighted point-accuracy term on mu specifically. The
    aux term closes the "inflate r instead of fixing mu" escape hatch:
    NB-NLL alone can be minimized by giving up on predicting a spike's mean
    correctly as long as r grows to make that spike less surprising: this
    term penalizes mu directly regardless of what r does, so that escape
    route no longer reduces the total loss. aux_weight=0.15 keeps this
    secondary to the main NB-NLL term rather than dominating it.

    `beta`/`r_reference` (defaults 0.0/10.0, exact no-op): passed through to
    negative_binomial_nll -- see its docstring for the full reasoning
    (adapting Seitzer et al.'s beta-NLL to this NB setting). Combinable
    with the aux term (both target the same escape hatch through different,
    complementary mechanisms) or usable alone (aux_weight=0.0, beta>0) to
    test whether beta-reweighting handles it without the auxiliary term at
    all -- worth comparing both ways rather than assuming either alone.
    """
    nll = negative_binomial_nll(mu, r, target, beta=beta, r_reference=r_reference)
    aux = weighted_point_loss(mu, target, weight_power=weight_power)
    return nll + aux_weight * aux

