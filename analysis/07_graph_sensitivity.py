"""
Graph-fitting sensitivity (reported in Section III-B and the Discussion of the paper).

Question: the lagged-correlation graph is fitted once on the full official training
period, which contains the internal-validation days that phase-1 epoch-budget selection
uses. Does that make phase-1 validation optimistic?

Test: for each city and seed, run phase 1 (train_one_run: internal_fit -> internal_val,
early stopping) twice, changing ONLY the correlation graph:
    "train"  : graph fitted on idx["train"]        (what the paper does; sees internal_val days)
    "ifit"   : graph fitted on idx["internal_fit"] (blind to internal_val days)
and compare internal-validation loss / MAE and the selected epoch budget.

Run from the repo root (needs Metro_Data/, or set METRO_DATA_DIR):
    python analysis/07_graph_sensitivity.py --city Hangzhou --seeds 5
    python analysis/07_graph_sensitivity.py --city Shanghai --seeds 5
    python analysis/07_graph_sensitivity.py --city Beijing  --seeds 5

Results are appended to analysis/results/graph_sensitivity_<city>.json after every run, so you
can stop and resume (finished (graph, seed) pairs are skipped). Training is real (10 runs per
city); on CPU it takes hours, on Apple-silicon MPS considerably less.

The reported results used the defaults below, which are the defaults in cahgtn/train.py
(patience 15, minimum 25 epochs, ceiling 100, learning rate 1e-3); they were not the
learning rate selected per city by notebook 01's phase-1 search. Check OUTPUT_BIAS_INIT too.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
                help="repo root (contains cahgtn/ and Metro_Data/)")
ap.add_argument("--city", required=True, choices=["Hangzhou", "Shanghai", "Beijing"])
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
ap.add_argument("--ceiling", type=int, default=100)
ap.add_argument("--patience", type=int, default=15)
ap.add_argument("--min-epochs", type=int, default=25)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--out-dir", default=None)
args = ap.parse_args()

ROOT = os.path.abspath(args.root)
sys.path.insert(0, ROOT)
from cahgtn.data_pipeline import (load_city, build_seasonal_profile, build_windows,
                                  _build_lagged_correlation_adjacency)
from cahgtn.model import CAHGTN
from cahgtn.train import WindowDataset, train_one_run, predict_all

if args.device == "auto":
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)
print("device:", device, flush=True)

out_dir = args.out_dir or os.path.join(ROOT, "analysis", "results")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"graph_sensitivity_{args.city}.json")
results = json.load(open(out_path)) if os.path.exists(out_path) else {}

data = load_city(os.environ.get("METRO_DATA_DIR", os.path.join(ROOT, "Metro_Data")), args.city)
ix = data.idx
N = data.num_stations
print(args.city, "stations", N, {k: len(v) for k, v in ix.items()}, flush=True)

# Phase-1 data: seasonal profile fitted on internal_fit only (as in the leak-free protocol).
prof = build_seasonal_profile(data, ix["internal_fit"])
fit_w = build_windows(data, ix["internal_fit"], prof)
val_w = build_windows(data, ix["internal_val"], prof)
print("internal_fit windows", fit_w[0].shape[0], " internal_val windows", val_w[0].shape[0], flush=True)

# Graphs
A_train = data.adjacency_corr                                   # fitted on idx["train"] (paper)
A_ifit_np, _ = _build_lagged_correlation_adjacency(data.flow[ix["internal_fit"]],
                                                   data.slot_id[ix["internal_fit"]], data.n_slots)
A_ifit = torch.tensor(np.asarray(A_ifit_np), dtype=torch.float32)
graphs = {"train": A_train, "ifit": A_ifit}
kept = []
for i in range(N):
    a = set(torch.nonzero(A_train[i] > 0).flatten().tolist()); b = set(torch.nonzero(A_ifit[i] > 0).flatten().tolist())
    kept.append(len(a & b) / max(len(a), 1))
print("fraction of full-train edges also in internal_fit graph:", round(float(np.mean(kept)), 3), flush=True)

# !! Check against your notebook: initial output bias target = mean (inflow, outflow) of the fitting data.
OUTPUT_BIAS_INIT = tuple(float(v) for v in data.flow[ix["internal_fit"]].mean(axis=(0, 1)))

def make_model():
    return CAHGTN(num_stations=N, hidden_dim=64, horizon=4, head_type="nb", graph_mode="full",
                  use_seasonal_residual=True, output_bias_init=OUTPUT_BIAS_INIT)

val_loader = DataLoader(WindowDataset(*val_w), batch_size=64, shuffle=False)

for seed in range(args.seeds):
    for gname in ["train", "ifit"]:
        key = f"{gname}|{seed}"
        if key in results:
            print("skip", key, flush=True); continue
        t0 = time.time()
        A = graphs[gname]
        model, best_epoch = train_one_run(
            make_model, fit_w, val_w, data.adjacency_physical, A, device, head_type="nb",
            epochs=args.ceiling, batch_size=32, lr=args.lr, patience=args.patience, seed=seed,
            min_epochs_before_stopping=args.min_epochs)
        model.eval()
        p, y = predict_all(model, val_loader, data.adjacency_physical.to(device), A.to(device), device, "nb")
        mae = float(np.abs(p["mu"] - y).mean())
        mae_h1 = float(np.abs(p["mu"][:, 0] - y[:, 0]).mean())
        results[key] = dict(graph=gname, seed=seed, best_epoch=int(best_epoch), val_mae_all_horizons=mae,
                            val_mae_15min=mae_h1, seconds=round(time.time() - t0))
        json.dump(results, open(out_path, "w"), indent=1)
        print(f"== {args.city} {key}: best_epoch={best_epoch} val_MAE(15min)={mae_h1:.3f} "
              f"val_MAE(all)={mae:.3f}  [{time.time() - t0:.0f}s]", flush=True)

# Summary: paired by seed
print("\nSUMMARY (internal_val; lower is better)")
rows = []
for s in range(args.seeds):
    if f"train|{s}" in results and f"ifit|{s}" in results:
        a, b = results[f"train|{s}"], results[f"ifit|{s}"]
        rows.append((a["val_mae_15min"], b["val_mae_15min"], a["best_epoch"], b["best_epoch"]))
if rows:
    r = np.array(rows)
    d = r[:, 1] - r[:, 0]
    print(f"  graph fitted on full train : 15-min val MAE {r[:,0].mean():.3f}, mean best epoch {r[:,2].mean():.1f}")
    print(f"  graph fitted on internal_fit: 15-min val MAE {r[:,1].mean():.3f}, mean best epoch {r[:,3].mean():.1f}")
    print(f"  paired diff (ifit - train): mean {d.mean():+.3f}, sd {d.std(ddof=1) if len(d) > 1 else float('nan'):.3f}, n={len(d)}")
    print("  If the diff is ~0 (relative to seed sd) the shared-graph optimism is negligible;"
          " a clearly positive diff means validation under the paper's graph was optimistic.")
