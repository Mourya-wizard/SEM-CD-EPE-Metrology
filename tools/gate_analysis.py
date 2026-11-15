"""Pre-registered gate checks for E13 and E14 on the VALIDATION set (dose 8, rep 0). See docs/ROADMAP.md.

  python tools/gate_analysis.py e13-diagnose <base_run> [--reg _regbias]
  python tools/gate_analysis.py e13-gate     <base_run> <new_run> [--reg _regbias]
  python tools/gate_analysis.py e14-gate     <base_run> <new_run> [--reg _regbias]
--reg selects the estimated-registration measurements: '' (CAD template, default) or '_regbias' (E12 adopted).
Needs val measurements <run>{reg}__learned and <run>_oraclereg__learned (python src/evaluate.py run-learned).
Writes results/roadmap_gates/<command>.json and prints it.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import evaluate as ev  # noqa: E402
import train as tr  # noqa: E402
from data_lithobench import load_config  # noqa: E402
from models import build_for_run  # noqa: E402


def merged_val(cfg, name, truth, dose):
    return ev.merged(cfg, ev.load_meas(cfg, "val", name)[lambda d: (d.rep == 0) & (d.dose == dose)], truth)


def bloom_offsets(cfg):
    """Per-tile bloom offset (px) of the cached validation tiles."""
    out = {}
    for s, t in ev.eval_tiles(cfg, "val"):
        z = np.load(os.path.join(cfg["paths"]["results"], "raw", "cache", "val", s, t + ".npz"))
        out[(s, t)] = json.loads(str(z["params"]))["bloom_offset_px"]
    return out


def diagnose(cfg, base, truth, dose):
    """E13 step 0: how much of the CD error (ideal registration, inside range) is a per-image offset?"""
    m = merged_val(cfg, f"{base}_oraclereg__learned", truth, dose)
    m = m[(m.kind == "cd") & (m.inside == True) & ~m.fail]  # noqa: E712
    off = bloom_offsets(cfg)
    m["delta"] = [off[(s, t)] for s, t in zip(m.set, m.tile)]
    key = ["set", "tile", "cond"]
    g = m.groupby(key)
    mi = g.err.transform("mean")
    total_var = float(m.err.var())
    within = float((m.err - mi).var())
    per_img = m.groupby(key).agg(mean_err=("err", "mean"), delta=("delta", "first"), n=("err", "size")).reset_index()
    r = float(np.corrcoef(per_img.mean_err, per_img.delta)[0, 1])
    slope = float(np.polyfit(per_img.delta, per_img.mean_err, 1)[0])
    share = 1 - within / total_var
    res = {"base": base, "n_gauges": int(len(m)), "n_images": int(len(per_img)), "error_var_nm2": total_var,
           "per_image_share": float(share), "pearson_r_error_vs_offset": r, "slope_nm_per_px": slope,
           "gate_share_ge_0.30": bool(share >= 0.30), "gate_abs_r_ge_0.5": bool(abs(r) >= 0.5)}
    res["proceed_to_training"] = res["gate_share_ge_0.30"] and res["gate_abs_r_ge_0.5"]
    return res


def offset_r2(cfg, run, n_per_set=30):
    ck = torch.load(os.path.join(ROOT, "checkpoints", run, "best.pt"), map_location="cuda", weights_only=False)
    model = build_for_run(cfg["model"], ck["run"]).cuda().eval()
    model.load_state_dict(ck["model"])
    sc, off = cfg["sim"], bloom_offsets(cfg)
    lim = cfg["sim"]["train"]["bloom_offset_px"][1]
    inside = [i for i, c in enumerate(ev.conditions(cfg)) if c["inside"]]
    pred, true = [], []
    for s in ("MetalSet", "ViaSet"):
        for t in [t for s2, t in ev.eval_tiles(cfg, "val") if s2 == s][:n_per_set]:
            d = ev.load_cache(os.path.join(cfg["paths"]["results"], "raw", "cache", "val", s, t + ".npz"))
            for c in inside[:3]:
                img = ev.eval_noisy(d["clean"][c], cfg["eval"]["selection_dose"], int(d["seed"]) * 100 + c, 0, sc)
                x = torch.as_tensor(img, device="cuda", dtype=torch.float32)[None, None]
                x = (x - x.mean()) / x.std().clamp(min=1e-6)
                with torch.no_grad():
                    pred.append(float(model(x, return_aux=True)[1][0]) * lim)
                true.append(off[(s, t)])
    pred, true = np.array(pred), np.array(true)
    return float(1 - ((pred - true) ** 2).sum() / ((true - true.mean()) ** 2).sum())


def e13_gate(cfg, base, new, truth, dose, reg):
    cd = lambda name, **kw: ev.accuracy_regression(merged_val(cfg, name, truth, dose).query("kind == 'cd' and inside == True"))  # noqa: E731
    ora_b, ora_n = cd(f"{base}_oraclereg__learned"), cd(f"{new}_oraclereg__learned")
    est_b, est_n = cd(f"{base}{reg}__learned"), cd(f"{new}{reg}__learned")
    stress = lambda name: ev.accuracy_regression(merged_val(cfg, name, truth, dose).query("kind == 'cd' and cname == 'stress_contrast'"))["rmse"]  # noqa: E731
    r2 = offset_r2(cfg, new)
    res = {"base": base, "new": new, "oracle_cd_tmu": [ora_b["tmu3s"], ora_n["tmu3s"]], "est_cd_tmu": [est_b["tmu3s"], est_n["tmu3s"]],
           "stress_contrast_rmse": [stress(f"{base}{reg}__learned"), stress(f"{new}{reg}__learned")], "offset_head_r2": r2}
    g = {"G1_oracle_tmu_drop_ge_0.30nm": ora_n["tmu3s"] <= ora_b["tmu3s"] - 0.30,
         "G2_est_tmu_not_worse_0.05": est_n["tmu3s"] <= est_b["tmu3s"] + 0.05,
         "G3_stress_rmse_not_worse_0.10": res["stress_contrast_rmse"][1] <= res["stress_contrast_rmse"][0] + 0.10,
         "G4_offset_r2_ge_0.5": r2 >= 0.5}
    res["gates"] = {k: bool(v) for k, v in g.items()}
    res["adopt_aux_head"] = bool(all(g.values()))
    return res


def e14_gate(cfg, base, new, truth, dose, reg):
    fig = next(os.path.join(cfg["paths"]["results"], d) for d in ("figures", "final") if os.path.isdir(os.path.join(cfg["paths"]["results"], d)))
    bias = {}
    for run in (base, new):
        df = pd.concat([pd.read_csv(os.path.join(fig, f"roughness_{run}_sigma{s}.csv")) for s in (1, 2)])
        df = df[df.dose == 64]
        bias[run] = float((100 * (df.ler_raw_3s / df.truth_ler_3s - 1)).abs().mean())
    s = {}
    for run in (base, new):
        sm = ev.summarize(cfg, ev.load_meas(cfg, "val", f"{run}{reg}__learned"), truth, run, n_boot=0)
        r = sm[(sm.kind == "cd") & (sm.dose == dose)].iloc[0]
        s[run] = {"cd_tmu3s": float(r.inside_tmu3s), "cd_prec3s": float(r.prec_3s)}
    res = {"base": base, "new": new, "abs_ler_bias_pct_64fr": [bias[base], bias[new]], "cd_tmu3s": [s[base]["cd_tmu3s"], s[new]["cd_tmu3s"]],
           "cd_prec3s_8fr": [s[base]["cd_prec3s"], s[new]["cd_prec3s"]]}
    g = {"G1_ler_bias_le_2.5pct": bias[new] <= 2.5, "G2_tmu_not_worse_0.10": s[new]["cd_tmu3s"] <= s[base]["cd_tmu3s"] + 0.10,
         "G3_precision_le_1.10x": s[new]["cd_prec3s"] <= 1.10 * s[base]["cd_prec3s"]}
    res["gates"] = {k: bool(v) for k, v in g.items()}
    res["adopt_spectrum_loss"] = bool(all(g.values()))
    return res


if __name__ == "__main__":
    cmd, args = sys.argv[1], [a for a in sys.argv[2:] if not a.startswith("--")]
    reg = sys.argv[sys.argv.index("--reg") + 1] if "--reg" in sys.argv else ""
    args = [a for a in args if a != reg or not reg]
    cfg = load_config()
    dose = cfg["eval"]["selection_dose"]
    truth = ev.load_truth(cfg, "val")
    res = {"e13-diagnose": lambda: diagnose(cfg, args[0], truth, dose), "e13-gate": lambda: e13_gate(cfg, args[0], args[1], truth, dose, reg),
           "e14-gate": lambda: e14_gate(cfg, args[0], args[1], truth, dose, reg)}[cmd]()
    d = os.path.join(cfg["paths"]["results"], "roadmap_gates")
    os.makedirs(d, exist_ok=True)
    json.dump(res, open(os.path.join(d, f"{cmd}_{'_'.join(args)}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
