"""
cahgtn/train.py
================================================================
Training loop for CA-HGTN (and the ablation variants). Early
stopping / model selection uses the mixed-weekday internal
validation set (see data_pipeline.py); final numbers are reported
on the official, date-anchored test set for comparability with the
literature. Supports multiple seeds so a paper table can report
mean +/- std rather than a single run.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .losses import combined_nb_loss, weighted_point_loss
from .evaluate import per_horizon_report


class WindowDataset(Dataset):
    def __init__(self, X, Y, TF, EXTRA, SEAS):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.TF = torch.from_numpy(TF)
        self.EXTRA = torch.from_numpy(EXTRA)
        self.SEAS = torch.from_numpy(SEAS)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.TF[idx], self.EXTRA[idx], self.SEAS[idx]


def run_epoch(model, loader, adj_phys, adj_corr, device, head_type, optimizer=None,
              adaptive_entropy_reg_coef: float = 0.0,
              nb_loss_beta: float = 0.0, nb_loss_r_reference: float = 10.0,
              nb_loss_aux_weight: float = 0.15):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, n_batches = 0.0, 0
    with torch.set_grad_enabled(training):
        for X, Y, TF, EXTRA, SEAS in loader:
            X, Y, TF, EXTRA, SEAS = X.to(device), Y.to(device), TF.to(device), EXTRA.to(device), SEAS.to(device)
            out = model(X, TF, EXTRA, SEAS, adj_phys, adj_corr)

            if head_type == "nb":
                loss = combined_nb_loss(out["mu"], out["r"], Y, aux_weight=nb_loss_aux_weight,
                                         beta=nb_loss_beta, r_reference=nb_loss_r_reference)
            else:
                loss = weighted_point_loss(out["pred"], Y)

            # OPT-IN, default-off: see CAHGTN.adaptive_graph_entropy_regularization
            # in model.py for what this is and why it's off by default. Only
            # CAHGTN has this method (GRUOnlyBaseline/GCNGRUBaseline don't have
            # an adaptive graph), so this is a no-op for those regardless.
            if adaptive_entropy_reg_coef > 0.0 and hasattr(model, "adaptive_graph_entropy_regularization"):
                reg = model.adaptive_graph_entropy_regularization()
                if reg is not None:
                    loss = loss + adaptive_entropy_reg_coef * reg

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def predict_all(model, loader, adj_phys, adj_corr, device, head_type):
    model.eval()
    outs = {"mu": [], "r": [], "pred": []}
    trues = []
    for X, Y, TF, EXTRA, SEAS in loader:
        X, Y, TF, EXTRA, SEAS = X.to(device), Y.to(device), TF.to(device), EXTRA.to(device), SEAS.to(device)
        out = model(X, TF, EXTRA, SEAS, adj_phys, adj_corr)
        if head_type == "nb":
            outs["mu"].append(out["mu"].cpu().numpy())
            outs["r"].append(out["r"].cpu().numpy())
        else:
            outs["pred"].append(out["pred"].cpu().numpy())
        trues.append(Y.cpu().numpy())

    result = {}
    if head_type == "nb":
        result["mu"] = np.concatenate(outs["mu"], axis=0)
        result["r"] = np.concatenate(outs["r"], axis=0)
    else:
        result["pred"] = np.concatenate(outs["pred"], axis=0)
    return result, np.concatenate(trues, axis=0)


def train_one_run(
    model_ctor,
    train_data,
    val_data,
    adj_physical,
    adj_corr,
    device,
    head_type: str = "nb",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 15,
    seed: int = 0,
    warmup_epochs: int = 5,
    min_epochs_before_stopping: int = 25,
    min_delta_rel: float = 0.0015,
    weight_decay: float = 1e-3,
    adaptive_entropy_reg_coef: float = 0.0,
    nb_loss_beta: float = 0.0,
    nb_loss_r_reference: float = 10.0,
    nb_loss_aux_weight: float = 0.15,
):
    """Trains with early stopping against `val_data`. IMPORTANT: `val_data`
    must not overlap with `train_data` in time, or "early stopping" measures
    near-in-sample fit rather than generalization and the resulting
    checkpoint selection is close to meaningless. Returns (model, best_epoch)
    -- `best_epoch` is needed by the leak-free refit pattern in
    `train_fixed_epochs` below: pick the epoch budget using a genuinely
    held-out split, then refit on ALL available training data for exactly
    that many epochs, with no validation signal that overlaps the refit data.

    `warmup_epochs` / `min_epochs_before_stopping` exist to prevent a
    specific failure mode: this architecture's output heads start far from
    real ridership scale (see `_init_softplus_output_bias` in model.py for
    the fix on the initialization side of this same problem), so without
    bias initialization the first gradient step can be huge and noisy; even
    with it, a small/noisy first few epochs can still look artificially good
    by chance. Without a floor, early stopping can trigger almost
    immediately on that noise, before the model has had a real chance to
    learn.

    `min_delta_rel` addresses a DIFFERENT failure mode, seen after the fix
    above started working: val_loss on a small validation set (~500-800
    windows here) is genuinely noisy epoch-to-epoch, not monotonic. Without a
    minimum-improvement threshold, ANY improvement -- even 0.001 -- resets
    the patience counter, so the loop can chase noise for 60-90+ epochs
    before genuinely plateauing, well past the point of real generalization
    benefit; the resulting epoch count then gets baked into the final refit,
    and a follow-up run showed this actively hurt test performance relative
    to an earlier (accidentally under-trained) run. Requiring a real,
    proportional improvement (default 0.3%) to count as "better" filters
    that noise-chasing while still allowing genuine, larger improvements
    through. `weight_decay` is also increased from an earlier default of
    1e-5 to 1e-3 -- more L2 regularization is a second, complementary lever
    against the same overfitting risk (~1300 overlapping training windows is
    genuinely small for this model's capacity).

    `adaptive_entropy_reg_coef` (default 0.0, i.e. off): see
    CAHGTN.adaptive_graph_entropy_regularization in model.py. A hypothesis
    for reducing the cross-seed instability seen in `no-physical` and
    `point head` in the 5-seed Hangzhou run, not a verified fix -- test
    before trusting it.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(WindowDataset(*train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(*val_data), batch_size=batch_size, shuffle=False)

    model = model_ctor().to(device)
    adj_phys = adj_physical.to(device)
    adj_c = adj_corr.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_state, best_epoch, patience_left = float("inf"), None, 0, patience
    patience_reference = float("inf")
    for epoch in range(1, epochs + 1):
        _set_lr_with_warmup(optimizer, epoch, lr, warmup_epochs, epochs)
        train_loss = run_epoch(model, train_loader, adj_phys, adj_c, device, head_type, optimizer,
                                adaptive_entropy_reg_coef=adaptive_entropy_reg_coef,
                                nb_loss_beta=nb_loss_beta, nb_loss_r_reference=nb_loss_r_reference,
                                nb_loss_aux_weight=nb_loss_aux_weight)
        val_loss = run_epoch(model, val_loader, adj_phys, adj_c, device, head_type, optimizer=None,
                              adaptive_entropy_reg_coef=adaptive_entropy_reg_coef,
                              nb_loss_beta=nb_loss_beta, nb_loss_r_reference=nb_loss_r_reference,
                              nb_loss_aux_weight=nb_loss_aux_weight)
        print(f"  seed={seed} epoch={epoch:03d} lr={optimizer.param_groups[0]['lr']:.2e} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # Always keep the single best checkpoint ever seen, regardless of min_delta_rel.
        if val_loss < best_val:
            best_val, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch

        # But only reset the patience clock on a MEANINGFUL improvement (see
        # min_delta_rel docstring above) -- this is what actually prevents
        # noise-chasing, decoupled from which checkpoint gets kept.
        if val_loss < patience_reference * (1 - min_delta_rel):
            patience_reference = val_loss
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0 and epoch >= min_epochs_before_stopping:
                print(f"  early stop at epoch {epoch} (best val_loss={best_val:.4f} at epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    return model, best_epoch


def search_patience_and_ceiling(
    model_ctor,
    internal_fit,
    internal_val,
    adj_physical,
    adj_corr,
    device,
    candidates: list,
    head_type: str = "nb",
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 0,
    min_delta_rel: float = 0.0015,
    weight_decay: float = 1e-3,
):
    """Searches over (patience, min_epochs_before_stopping, ceiling) candidate
    configs, selecting the best one by VALIDATION MAE on internal_val --
    NEVER official_test. This is the fix for the leakage pattern identified
    this session: choosing a patience/epoch-ceiling config by watching
    official_test MAE improve across successive attempts (30 -> 150 -> 250)
    is structurally the same problem as early-stopping directly against the
    test set, even though each individual training run inside that process
    was itself leak-free -- the choice of WHICH config to settle on was the
    leaky step, not any single run. This function makes that choice against
    internal_val instead, the same way Phase 1's learning-rate search already
    works, so the winning config can go into the official refit+test
    evaluation exactly once, with no asterisk needed.

    `candidates`: list of (patience, min_epochs_before_stopping, ceiling)
    tuples. Deliberately does NOT touch official_train or official_test --
    only internal_fit/internal_val, exactly like the existing LR search.
    Uses a single seed for the search itself (consistent with how the LR
    search also doesn't multi-seed) -- run the full N_SEEDS official protocol
    only for the winning config afterward.

    Returns (results, best) where results is a list of dicts (one per
    candidate, for transparency/reporting -- report ALL of them, not just the
    winner, so the search itself is auditable) and best is the single
    dict with the lowest val_mae.
    """
    val_loader = DataLoader(WindowDataset(*internal_val), batch_size=batch_size, shuffle=False)
    adj_phys_dev = adj_physical.to(device)
    adj_c_dev = adj_corr.to(device)

    results = []
    for patience, min_epochs, ceiling in candidates:
        print(f"\n=== Searching: patience={patience}, min_epochs_before_stopping={min_epochs}, "
              f"ceiling={ceiling} ===")
        model, chosen_epoch = train_one_run(
            model_ctor, internal_fit, internal_val, adj_physical, adj_corr, device,
            head_type=head_type, epochs=ceiling, batch_size=batch_size,
            lr=lr, patience=patience, seed=seed,
            min_epochs_before_stopping=min_epochs,
            min_delta_rel=min_delta_rel, weight_decay=weight_decay,
        )
        val_mae = _quick_test_mae(model, val_loader, adj_phys_dev, adj_c_dev, device, head_type)
        print(f"  -> chosen_epoch={chosen_epoch}, VALIDATION MAE={val_mae:.3f} (internal_val, not official_test)")
        results.append({
            "patience": patience, "min_epochs_before_stopping": min_epochs, "ceiling": ceiling,
            "chosen_epoch": chosen_epoch, "val_mae": val_mae,
        })

    best = min(results, key=lambda r: r["val_mae"])
    print(f"\n{'=' * 70}")
    print("SEARCH COMPLETE -- all candidates, ranked by validation MAE:")
    for r in sorted(results, key=lambda r: r["val_mae"]):
        marker = " <-- BEST" if r is best else ""
        print(f"  patience={r['patience']:>4} min_epochs={r['min_epochs_before_stopping']:>4} "
              f"ceiling={r['ceiling']:>4}  val_mae={r['val_mae']:.3f}{marker}")
    print(f"\nBest by validation: patience={best['patience']}, "
          f"min_epochs_before_stopping={best['min_epochs_before_stopping']}, ceiling={best['ceiling']}")
    print("Use THIS config for the one-time official 5-seed refit+test evaluation --")
    print("official_test has not been touched by this search at all.")
    return results, best


def _set_lr_with_warmup(optimizer, epoch: int, base_lr: float, warmup_epochs: int, total_epochs: int):
    """Linear warmup from 10% of base_lr up to base_lr over `warmup_epochs`,
    then cosine decay for the remainder. Implemented as a direct per-epoch LR
    assignment (rather than composing multiple torch schedulers) so the
    schedule is easy to read and to unit-test independent of torch specifics."""
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        lr = base_lr * (0.1 + 0.9 * epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
    for g in optimizer.param_groups:
        g["lr"] = lr



def train_fixed_epochs(
    model_ctor,
    train_data,
    adj_physical,
    adj_corr,
    device,
    head_type: str = "nb",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    warmup_epochs: int = 5,
    weight_decay: float = 1e-3,
    adaptive_entropy_reg_coef: float = 0.0,
    nb_loss_beta: float = 0.0,
    nb_loss_r_reference: float = 10.0,
    nb_loss_aux_weight: float = 0.15,
):
    """Trains for exactly `epochs` epochs on ALL of `train_data`, with NO
    validation loop and NO early stopping -- i.e. no signal that could leak
    if `train_data` is your full official training set. Use this for the
    FINAL refit after `epochs` has already been chosen via a proper,
    non-overlapping validation split (see notebook 01, Phase 2): pick the
    epoch budget with `train_one_run` on a genuinely held-out split, then
    call this function on the full training set for that many epochs. Since
    there's no legitimate held-out signal available during this final fit,
    it simply returns the LAST epoch's weights rather than a "best" epoch
    (there's nothing valid to select "best" against here).

    `adaptive_entropy_reg_coef`/`nb_loss_beta`/`nb_loss_r_reference`: pass
    the SAME values used in the `train_one_run` call that selected `epochs`,
    for consistency between the two phases -- see model.py's
    CAHGTN.adaptive_graph_entropy_regularization and losses.py's
    negative_binomial_nll."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(WindowDataset(*train_data), batch_size=batch_size, shuffle=True)

    model = model_ctor().to(device)
    adj_phys = adj_physical.to(device)
    adj_c = adj_corr.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        _set_lr_with_warmup(optimizer, epoch, lr, min(warmup_epochs, max(1, epochs // 2)), epochs)
        train_loss = run_epoch(model, train_loader, adj_phys, adj_c, device, head_type, optimizer,
                                adaptive_entropy_reg_coef=adaptive_entropy_reg_coef,
                                nb_loss_beta=nb_loss_beta, nb_loss_r_reference=nb_loss_r_reference,
                                nb_loss_aux_weight=nb_loss_aux_weight)
        print(f"  [refit] seed={seed} epoch={epoch:03d}/{epochs} "
              f"lr={optimizer.param_groups[0]['lr']:.2e} train_loss={train_loss:.4f}")

    return model


@torch.no_grad()
def _quick_test_mae(model, test_loader, adj_phys, adj_c, device, head_type) -> float:
    preds, trues = predict_all(model, test_loader, adj_phys, adj_c, device, head_type)
    return per_horizon_report(preds, trues)[0]["MAE"]  # 15-min horizon


def diagnostic_train_with_test_trajectory(
    model_ctor,
    train_data,
    test_data,
    adj_physical,
    adj_corr,
    device,
    head_type: str = "nb",
    epochs: int = 300,
    eval_every: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    warmup_epochs: int = 5,
    weight_decay: float = 1e-3,
):
    """DIAGNOSTIC ONLY -- not part of this project's official training or
    evaluation protocol, and its output must never be used to select an
    epoch count, hyperparameter, or checkpoint for anything reported
    elsewhere in this project. Every other training function here is
    deliberately test-blind; this one is not, on purpose, for one specific
    question: trains for a long, FIXED epoch budget (no early stopping,
    unlike train_one_run) and periodically evaluates on the TEST set
    (which train_fixed_epochs and train_one_run never touch) to trace how
    test performance evolves over long, unstopped training.

    This directly tests the "PB-GRU's fixed 350-epoch schedule might be
    overfitting to a test period sitting immediately after training"
    hypothesis this project has discussed but never actually checked:
    test MAE continuing to fall throughout training is evidence against
    it (the model genuinely benefits from more training); test MAE
    falling then rising while train loss keeps falling is the direct,
    checkable signature of overfitting.

    Uses test-set predictions during training, which every other function
    in this project deliberately avoids -- legitimate ONLY as an
    observational diagnostic (answering "what does the curve look like"),
    not as a new way to pick a stopping point (answering "what's the best
    checkpoint"). Keep this function's output out of {CITY}_predictions.pkl
    and out of notebook 02's aggregation entirely -- it answers a different
    question than everything else in this project and should stay visibly
    separate, not quietly blended in as if it were another ablation.

    Returns (model, trajectory) where trajectory is a list of dicts with
    epoch/train_loss/test_mae_15min, recorded at epoch 1, every
    `eval_every` epochs, and the final epoch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(WindowDataset(*train_data), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(WindowDataset(*test_data), batch_size=batch_size, shuffle=False)

    model = model_ctor().to(device)
    adj_phys = adj_physical.to(device)
    adj_c = adj_corr.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    trajectory = []
    for epoch in range(1, epochs + 1):
        _set_lr_with_warmup(optimizer, epoch, lr, warmup_epochs, epochs)
        train_loss = run_epoch(model, train_loader, adj_phys, adj_c, device, head_type, optimizer)

        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            test_mae = _quick_test_mae(model, test_loader, adj_phys, adj_c, device, head_type)
            trajectory.append({"epoch": epoch, "train_loss": train_loss, "test_mae_15min": test_mae})
            print(f"  [diagnostic] epoch={epoch:04d}/{epochs} train_loss={train_loss:.4f} "
                  f"test_MAE(15min)={test_mae:.2f}")

    return model, trajectory


def diagnostic_summarize_reproducibility(trajectories: list, official_std_reference: float = 0.33) -> dict:
    """DIAGNOSTIC ONLY, same status as diagnostic_train_with_test_trajectory
    above -- this is analysis of that function's output across multiple
    seeds, not a new official result. Answers a specific, narrower question
    than "does long training overfit": is the late-training convergence
    point a genuine, reproducible property of this architecture on this
    dataset, or an artifact of one particular random initialization? A
    single trajectory settling into a tight, low-variance late-training
    window (as observed for CA-HGTN full, seed 0: epochs 250-300, mean
    26.19, spread 0.32) only establishes that THAT RUN converged -- it
    says nothing about whether a different seed would converge to the same
    place. This checks that directly, across seeds.

    Uses the SAME windowed late-training comparison (last ~20% of epochs)
    as the single-trajectory diagnostic, applied independently to each
    seed's trajectory, then compares the resulting per-seed late-window
    means TO EACH OTHER -- not to a single reference point, for the same
    single-point-fragility reason documented in
    diagnostic_train_with_test_trajectory's own history (see PAPER_PLAN.md
    for the bug this project already caught and fixed once from exactly
    that kind of comparison).

    `official_std_reference` (default 0.33, this project's own CA-HGTN
    (full) official 5-seed MAE std): the natural yardstick for "is this
    seed-to-seed spread unusual" -- if the late-window means' own
    cross-seed std is similar to or smaller than ordinary seed variance
    elsewhere in this project, that's consistent with a real, stable
    convergence property; if it's much larger, the apparent convergence
    point is itself substantially seed-dependent, undermining the case
    that any single number here (e.g. "~26 MAE") should be trusted as a
    reliable property of the method rather than one lucky run.

    Returns per-seed late-window means/spreads and best-epoch/best-MAE,
    plus the cross-seed mean and std of the late-window means -- print the
    returned dict's contents directly; deliberately does not auto-generate
    a verdict string the way the single-trajectory function's notebook
    cell used to (that verdict logic was wrong once already on real data --
    see PAPER_PLAN.md -- and a 5-point cross-seed comparison deserves a
    human looking at the actual numbers, not another auto-generated
    one-liner)."""
    import numpy as np

    per_seed = []
    for trajectory in trajectories:
        maes = [t["test_mae_15min"] for t in trajectory]
        n = len(trajectory)
        window = max(1, n // 5)
        late_maes = maes[-window:]
        late_mean = sum(late_maes) / len(late_maes)
        late_spread = max(late_maes) - min(late_maes)
        best_idx = maes.index(min(maes))
        per_seed.append({
            "late_window_mean": late_mean,
            "late_window_spread": late_spread,
            "best_epoch": trajectory[best_idx]["epoch"],
            "best_mae": maes[best_idx],
            "final_epoch": trajectory[-1]["epoch"],
            "final_mae": maes[-1],
        })

    late_window_means = np.array([s["late_window_mean"] for s in per_seed])
    best_epochs = [s["best_epoch"] for s in per_seed]

    return {
        "per_seed": per_seed,
        "late_window_mean_across_seeds": float(late_window_means.mean()),
        "late_window_std_across_seeds": float(late_window_means.std()),
        "late_window_means_list": late_window_means.tolist(),
        "best_epochs_list": best_epochs,
        "official_std_reference": official_std_reference,
    }
