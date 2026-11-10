import os, sys, pickle, numpy as np
from pathlib import Path
ROOT = Path(os.environ.get("CAHGTN_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("METRO_DATA_DIR", ROOT / "Metro_Data"))
CACHE = ROOT / "cache"
RESULTS = ROOT / "analysis" / "results"
CACHE.mkdir(exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

def find(name, *subdirs):
    """Locate a data file: cache/, repo root, then any extra subdirs (relative to the root)."""
    for d in [CACHE, ROOT] + [ROOT / s for s in subdirs]:
        if (d / name).exists():
            return str(d / name)
    raise FileNotFoundError(f"{name} not found in cache/, the repo root, or {list(subdirs)}. "
                            f"Large files are in the Zenodo archive; see the README.")
from cahgtn.evaluate import (ensemble_predictions, find_calibration_scale, apply_calibration_scale,
                             nb_calibration, point_metrics)

H = 4
def load_city(city):
    """returns dict variant -> list of per-seed pred dicts; trues (n,4,N,2)"""
    out = {}
    if city == "Hangzhou":
        d = pickle.load(open(find("Hangzhou_predictions.pkl"), "rb"))
        ap = d["all_predictions"]
        for k in ["CA-HGTN (full)", "GCN+GRU baseline", "CA-HGTN (point head)", "CA-HGTN (single-graph)"]:
            out[k] = [p for p, t in ap[k]]
        trues = np.asarray(ap["CA-HGTN (full)"][0][1], dtype=np.float32)
        out = {k: [{f: np.asarray(v, dtype=np.float32) for f, v in p.items()} for p in lst] for k, lst in out.items()}
        return out, trues
    z = np.load(find(f"{city}_reduced.npz"))
    keys = list(z.keys())
    variants = sorted({k.split("|")[0] for k in keys if "|" in k and not k.startswith("trues")})
    for v in variants:
        seeds = sorted({int(k.split("|")[1]) for k in keys if k.startswith(v + "|")})
        lst = []
        for s in seeds:
            d = {k.split("|")[2]: z[k] for k in keys if k.startswith(f"{v}|{s}|")}
            lst.append(d)
        out[v] = lst
    trues = z["trues|0"]
    return out, trues
