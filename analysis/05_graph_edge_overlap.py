import json, numpy as np, torch
from common import *
from cahgtn.data_pipeline import load_city as load_raw, _build_lagged_correlation_adjacency
out = {}
for city in ["Hangzhou", "Shanghai", "Beijing"]:
    data = load_raw(str(DATA_DIR), city)
    ix = data.idx
    A_full = data.adjacency_corr.numpy()
    A_ifit, lag_ifit = _build_lagged_correlation_adjacency(data.flow[ix["internal_fit"]], data.slot_id[ix["internal_fit"]], data.n_slots)
    A_ifit = np.asarray(A_ifit)
    N = A_full.shape[0]
    # incoming edges: column or row? determine by non-zero counts
    nz_r = (A_full > 0).sum(1); nz_c = (A_full > 0).sum(0)
    ax = 1 if len(set(nz_r.tolist())) == 1 else 0   # axis along which each node has a fixed number (top-k) of edges
    S_full = [set(np.where(A_full[i if ax == 1 else slice(None), slice(None) if ax == 1 else i] > 0)[0].tolist()) for i in range(N)]
    S_ifit = [set(np.where(A_ifit[i if ax == 1 else slice(None), slice(None) if ax == 1 else i] > 0)[0].tolist()) for i in range(N)]
    k = np.mean([len(s) for s in S_full])
    jacc = np.mean([len(a & b) / max(len(a | b), 1) for a, b in zip(S_full, S_ifit)])
    frac_kept = np.mean([len(a & b) / max(len(a), 1) for a, b in zip(S_full, S_ifit)])
    # weight-level correlation between graphs
    flat_corr = float(np.corrcoef(A_full.ravel(), A_ifit.ravel())[0, 1])
    out[city] = dict(N=int(N), nnz_per_node=float(k), edge_set_overlap_frac=float(frac_kept), jaccard=float(jacc),
                     weight_corr=flat_corr, n_train_steps=int(len(ix["train"])), n_ifit_steps=int(len(ix["internal_fit"])), axis=ax)
    print(city, out[city], flush=True)
json.dump(out, open(RESULTS / "graph_edge_overlap.json", "w"), indent=1)
