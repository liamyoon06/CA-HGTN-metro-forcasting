import json, time
from common import *
res = {}
t0 = time.time()
for city in ["Hangzhou", "Beijing", "Shanghai"]:
    d, trues = load_city(city)
    full = d["CA-HGTN (full)"]
    ens = ensemble_predictions(full)
    n = trues.shape[0]
    R = res[city] = {}
    for h in range(H):
        y = trues[:, h]
        mu, r = ens["mu"][:, h], ens["r"][:, h]
        out = {}
        # per-seed uncorrected on full test (mean over seeds)
        ps = [nb_calibration(s["mu"][:, h], s["r"][:, h], y, level=0.90) for s in full]
        ps50 = [nb_calibration(s["mu"][:, h], s["r"][:, h], y, level=0.50) for s in full]
        out["seedmean_cov90"] = float(np.mean([p["coverage@90"] for p in ps]))
        out["seedmean_wid90"] = float(np.mean([p["width@90"] for p in ps]))
        out["seedmean_cov50"] = float(np.mean([p["coverage@50"] for p in ps50]))
        out["seedmean_wid50"] = float(np.mean([p["width@50"] for p in ps50]))
        # ensemble full-test uncorrected
        e90 = nb_calibration(mu, r, y, level=0.90); e50 = nb_calibration(mu, r, y, level=0.50)
        out["ens_full_cov90"], out["ens_full_wid90"] = e90["coverage@90"], e90["width@90"]
        out["ens_full_cov50"], out["ens_full_wid50"] = e50["coverage@50"], e50["width@50"]
        # --- protocol 1: odd fit / even report (the paper's)
        def protocol(fit, rep, tag):
            c = find_calibration_scale(mu[fit], r[fit], y[fit], target_coverage=0.90)
            b90 = nb_calibration(mu[rep], r[rep], y[rep], level=0.90)
            a90 = nb_calibration(mu[rep], apply_calibration_scale(r[rep], c), y[rep], level=0.90)
            b50 = nb_calibration(mu[rep], r[rep], y[rep], level=0.50)
            a50 = nb_calibration(mu[rep], apply_calibration_scale(r[rep], c), y[rep], level=0.50)
            out[tag] = dict(c=c, cov90_b=b90["coverage@90"], cov90_a=a90["coverage@90"],
                            wid90_b=b90["width@90"], wid90_a=a90["width@90"],
                            cov50_b=b50["coverage@50"], cov50_a=a50["coverage@50"],
                            wid50_b=b50["width@50"], wid50_a=a50["width@50"],
                            nll_b=b90["NB_NLL"], nll_a=a90["NB_NLL"], n_rep=int(rep.sum()))
        idx = np.arange(n)
        protocol(idx % 2 == 1, idx % 2 == 0, "oddeven")
        # --- protocol 2: chronological first half fit -> second half report, and reverse
        half = n // 2
        protocol(idx < half, idx >= half, "chrono_fwd")
        protocol(idx >= half, idx < half, "chrono_rev")
        print(city, h, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["oddeven"].items()}, round(time.time()-t0), flush=True)
        # --- protocol 3: 5-block contiguous cross-fit, pooled
        blocks = np.array_split(idx, 5)
        covs90, covs50, wb90, wa90, wb50, wa50, cs = [], [], [], [], [], [], []
        for b in blocks:
            rep = np.zeros(n, bool); rep[b] = True
            c = find_calibration_scale(mu[~rep], r[~rep], y[~rep], target_coverage=0.90)
            b90 = nb_calibration(mu[rep], r[rep], y[rep], 0.90); a90 = nb_calibration(mu[rep], r[rep]/c, y[rep], 0.90)
            b50 = nb_calibration(mu[rep], r[rep], y[rep], 0.50); a50 = nb_calibration(mu[rep], r[rep]/c, y[rep], 0.50)
            w = rep.sum()
            covs90.append((a90["coverage@90"], w)); covs50.append((a50["coverage@50"], w))
            wb90.append((b90["width@90"], w)); wa90.append((a90["width@90"], w))
            wb50.append((b50["width@50"], w)); wa50.append((a50["width@50"], w)); cs.append(c)
        wavg = lambda L: float(sum(v*w for v, w in L)/sum(w for _, w in L))
        out["block5"] = dict(cs=cs, cov90_a=wavg(covs90), cov50_a=wavg(covs50), wid90_b=wavg(wb90), wid90_a=wavg(wa90),
                             wid50_b=wavg(wb50), wid50_a=wavg(wa50), blockcov90=[v for v, _ in covs90])
        R[h] = out
    json.dump(res, open(RESULTS / "calibration_test.json", "w"), indent=1)
print("done", time.time()-t0)
