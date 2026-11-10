import sys, time, json, numpy as np, torch
from torch.utils.data import DataLoader
from common import *
from cahgtn.data_pipeline import load_city as load_raw, build_seasonal_profile, build_windows
from cahgtn.model import CAHGTN
from cahgtn.train import WindowDataset, predict_all
torch.set_num_threads(2)
device = torch.device("cpu")
city = sys.argv[1]
t0 = time.time()
data = load_raw(str(DATA_DIR), city)
print(city, "raw loaded", data.flow.shape, {k: (len(v), str(data.time_index[v[0]])[:10], str(data.time_index[v[-1]])[:10]) for k, v in data.idx.items()}, round(time.time()-t0), flush=True)
npz = np.load(find(f"{city}_windows.npz"))
adj_c_npz = npz["adjacency_corr"]; adj_p_npz = npz["adjacency_physical"]
print("adj_corr match npz:", np.allclose(data.adjacency_corr.numpy(), adj_c_npz, atol=1e-6), "adj_phys match:", np.allclose(data.adjacency_physical.numpy(), adj_p_npz))
prof = build_seasonal_profile(data, data.idx["train"])
Xv = build_windows(data, data.idx["official_val"], prof)
Xt = build_windows(data, data.idx["test"], prof)
print("val windows", Xv[0].shape, "test windows", Xt[0].shape, "test X matches npz:", np.allclose(Xt[0], npz["X_test"]), "SEAS match:", np.allclose(Xt[4], npz["SEAS_test"]), flush=True)
N = data.num_stations
adj_p = torch.tensor(adj_p_npz); adj_c = torch.tensor(adj_c_npz)

def run(split, seed, n_limit=None):
    sd = torch.load(find(f"{city}_CA-HGTN_'full'_seed{seed}.pt", f"weights/{city}"), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    m = CAHGTN(num_stations=N, hidden_dim=64, horizon=4, head_type="nb", graph_mode="full", use_seasonal_residual=True)
    m.load_state_dict(sd); m.eval()
    tup = tuple(a[:n_limit] if n_limit else a for a in split)
    p, y = predict_all(m, DataLoader(WindowDataset(*tup), batch_size=64, shuffle=False), adj_p, adj_c, device, "nb")
    return p, y

# sanity: seed0 test predictions vs saved pickle (first 64 windows)
d, trues = load_city_saved = None, None
saved, trues_saved = load_city(city)
p0, y0 = run(Xt, 0, n_limit=64)
ref = saved["CA-HGTN (full)"][0]
print("sanity seed0 mu max|diff| =", float(np.abs(p0["mu"]-ref["mu"][:64]).max()), " r max rel diff =", float(np.max(np.abs(p0["r"]-ref["r"][:64])/ref["r"][:64])), " trues match:", np.allclose(y0, trues_saved[:64]), round(time.time()-t0), flush=True)

vals = []
for s in range(5):
    p, y = run(Xv, s); vals.append(p); yv = y
    print("val seed", s, "done", round(time.time()-t0), flush=True)
np.savez_compressed(CACHE / f"valpreds_{city}.npz", yv=yv, **{f"mu{s}": vals[s]["mu"] for s in range(5)}, **{f"r{s}": vals[s]["r"] for s in range(5)})

ev = ensemble_predictions(vals)
et = ensemble_predictions(saved["CA-HGTN (full)"])
out = {"n_val": int(yv.shape[0]), "val_days": [str(data.time_index[data.idx["official_val"][0]]), str(data.time_index[data.idx["official_val"][-1]])]}
for h in range(4):
    mu_v, r_v, y_v = ev["mu"][:, h], ev["r"][:, h], yv[:, h]
    mu_t, r_t, y_t = et["mu"][:, h], et["r"][:, h], trues_saved[:, h]
    c = find_calibration_scale(mu_v, r_v, y_v, target_coverage=0.90)
    v90 = nb_calibration(mu_v, r_v, y_v, 0.90)
    b90 = nb_calibration(mu_t, r_t, y_t, 0.90); a90 = nb_calibration(mu_t, r_t / c, y_t, 0.90)
    b50 = nb_calibration(mu_t, r_t, y_t, 0.50); a50 = nb_calibration(mu_t, r_t / c, y_t, 0.50)
    out[h] = dict(c_val=c, val_cov90_before=v90["coverage@90"], val_wid90_before=v90["width@90"],
                  test_cov90_b=b90["coverage@90"], test_cov90_a=a90["coverage@90"],
                  test_wid90_b=b90["width@90"], test_wid90_a=a90["width@90"],
                  test_cov50_b=b50["coverage@50"], test_cov50_a=a50["coverage@50"],
                  test_wid50_b=b50["width@50"], test_wid50_a=a50["width@50"],
                  test_nll_b=b90["NB_NLL"], test_nll_a=a90["NB_NLL"],
                  val_mae=float(np.abs(mu_v-y_v).mean()), test_mae=float(np.abs(mu_t-y_t).mean()))
    print(city, "h", h+1, {k: round(v, 3) for k, v in out[h].items()}, round(time.time()-t0), flush=True)
json.dump(out, open(RESULTS / f"calibration_valfit_{city}.json", "w"), indent=1)
print("done", round(time.time()-t0))
