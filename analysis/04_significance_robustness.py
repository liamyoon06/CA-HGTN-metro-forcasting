import json, time
from scipy import stats
from common import *
from cahgtn.significance import diebold_mariano_test, bootstrap_relative_improvement

def win_err(pred, true):  # per-window mean abs error (matches significance._aggregate_window_errors, absolute)
    return np.abs(pred - true).reshape(pred.shape[0], -1).mean(axis=1)

def dm_bartlett(d, B, hln_h=1):
    """Bartlett-kernel HAC DM with bandwidth B (weights 1-k/B, k=1..B-1); B=1 -> iid."""
    T = len(d); db = d.mean(); g0 = np.mean((d-db)**2); v = g0
    for k in range(1, B):
        v += 2*(1-k/B)*np.mean((d[k:]-db)*(d[:-k]-db))
    v = max(v, 0)/T
    st = db/np.sqrt(v) if v > 0 else 0.0
    hln = np.sqrt(max((T+1-2*hln_h+hln_h*(hln_h-1)/T)/T, 0))
    st *= hln
    return float(st), float(2*(1-stats.t.cdf(abs(st), df=T-1)))

def acf(d, k):
    x = d-d.mean(); return float(np.sum(x[k:]*x[:-k])/np.sum(x*x))

def mbb_ci(e_new, e_base, L, nboot=5000, seed=0):
    n = len(e_new); rng = np.random.RandomState(seed)
    nb = int(np.ceil(n/L)); imp = np.empty(nboot)
    starts_max = n-L
    cs_new = np.concatenate([[0], np.cumsum(e_new)]); cs_base = np.concatenate([[0], np.cumsum(e_base)])
    for i in range(nboot):
        st = rng.randint(0, starts_max+1, size=nb)
        sn = (cs_new[st+L]-cs_new[st]).sum(); sb = (cs_base[st+L]-cs_base[st]).sum()
        imp[i] = (sb-sn)/sb*100
    lo, hi = np.percentile(imp, [2.5, 97.5])
    return float(lo), float(hi)

res = {}
t0 = time.time()
for city in ["Hangzhou", "Beijing", "Shanghai"]:
    d, trues = load_city(city)
    E = {name: ensemble_predictions(d[name]) for name in d}
    R = res[city] = {}
    comps = {"gcn": ("CA-HGTN (full)", "GCN+GRU baseline"),
             "pointhead": ("CA-HGTN (point head)", "CA-HGTN (full)"),
             "single": ("CA-HGTN (single-graph)", "CA-HGTN (full)")}
    for cname, (a, b) in comps.items():
        R[cname] = {}
        ka = "mu" if "mu" in E[a] else "pred"; kb = "mu" if "mu" in E[b] else "pred"
        for h in range(H):
            y = trues[:, h]; pa = E[a][ka][:, h]; pb = E[b][kb][:, h]
            ea, eb = win_err(pa, y), win_err(pb, y)
            d_ = ea-eb
            o = {"mae_a": float(ea.mean()), "mae_b": float(eb.mean()),
                 "imp_pct": float((eb.mean()-ea.mean())/eb.mean()*100), "n": len(ea),
                 "acf1": acf(d_, 1), "acf6": acf(d_, 6), "acf12": acf(d_, 12), "acf36": acf(d_, 36)}
            # paper's original quantities
            o["paper_dm"] = diebold_mariano_test(pa, pb, y, loss="absolute", h=h+1)["p_value"]
            bs = bootstrap_relative_improvement(pa, pb, y, loss="absolute", n_bootstrap=5000, ci=0.95, seed=0)
            o["iid_ci"] = [bs["ci_low"], bs["ci_high"]]
            # block bootstrap CIs
            o["mbb"] = {str(L): mbb_ci(ea, eb, L) for L in (6, 12, 36, 73)}
            # longer-lag DM
            o["dm_long"] = {str(B): dm_bartlett(d_, B)[1] for B in (6, 12, 37, 74)}
            o["dm_long_stat"] = {str(B): dm_bartlett(d_, B)[0] for B in (6, 12, 37, 74)}
            R[cname][h] = o
        print(city, cname, [(round(R[cname][h]["imp_pct"], 2), round(R[cname][h]["paper_dm"], 4)) for h in range(H)], round(time.time()-t0), flush=True)
    json.dump(res, open(RESULTS / "significance_robustness.json", "w"), indent=1)
print("done", time.time()-t0)
