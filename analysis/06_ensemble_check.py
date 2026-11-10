import json, numpy as np
from scipy.stats import nbinom
from common import *
from cahgtn.evaluate import _nb_scipy_params
out = {}
for city in ["Hangzhou", "Beijing", "Shanghai"]:
    d, trues = load_city(city); full = d["CA-HGTN (full)"]; ens = ensemble_predictions(full)
    out[city] = {}
    for h in range(4):
        y = trues[:, h]
        res = {}
        for level in (0.90, 0.50):
            a = (1-level)/2
            # exact mixture coverage: covered iff F(y) >= a and F(y-1) < 1-a
            Fy = np.zeros_like(y, dtype=np.float64); Fy1 = np.zeros_like(Fy)
            for s in full:
                n, p = _nb_scipy_params(s["mu"][:, h].astype(np.float64), s["r"][:, h].astype(np.float64))
                Fy += nbinom.cdf(y, n, p)/len(full); Fy1 += nbinom.cdf(y-1, n, p)/len(full)
            cov_mix = float(((Fy >= a) & (Fy1 < 1-a)).mean())
            n, p = _nb_scipy_params(ens["mu"][:, h].astype(np.float64), ens["r"][:, h].astype(np.float64))
            lo = nbinom.ppf(a, n, p); hi = nbinom.ppf(1-a, n, p)
            cov_mm = float(((y >= lo) & (y <= hi)).mean())
            res[f"mix{int(level*100)}"] = cov_mix; res[f"mm{int(level*100)}"] = cov_mm
        out[city][h] = res
        print(city, h+1, res, flush=True)
json.dump(out, open(RESULTS / "ensemble_check.json", "w"), indent=1)
