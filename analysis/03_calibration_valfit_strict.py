import json, numpy as np
from common import *
out = {}
for city in ["Hangzhou", "Shanghai"]:
    z = np.load(CACHE / f"valpreds_{city}.npz"); yv = z["yv"]
    seeds = [{"mu": z[f"mu{s}"], "r": z[f"r{s}"]} for s in range(5)]
    ev = ensemble_predictions(seeds)
    saved, trues = load_city(city); et = ensemble_predictions(saved["CA-HGTN (full)"])
    keep = np.arange(len(yv)) < len(yv) - 3     # drop last 3 val windows whose targets reach into the test period
    out[city] = {}
    for h in range(4):
        c = find_calibration_scale(ev["mu"][keep, h], ev["r"][keep, h], yv[keep, h], target_coverage=0.90)
        a90 = nb_calibration(et["mu"][:, h], et["r"][:, h]/c, trues[:, h], 0.90)
        a50 = nb_calibration(et["mu"][:, h], et["r"][:, h]/c, trues[:, h], 0.50)
        out[city][h] = dict(c=c, cov90_a=a90["coverage@90"], wid90_a=a90["width@90"], cov50_a=a50["coverage@50"], wid50_a=a50["width@50"], n=int(keep.sum()))
        print(city, h+1, {k: round(v, 3) for k, v in out[city][h].items()}, flush=True)
json.dump(out, open(RESULTS / "calibration_valfit_strict.json", "w"), indent=1)
