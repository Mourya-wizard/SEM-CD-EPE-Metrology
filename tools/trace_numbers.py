"""Save two diagnostics quoted in the README to a results file (previously only logged):
  1. registration accuracy against a template of the PRINTED pattern (truth contour) vs the CAD template,
     on 40 validation crops at 2 and 64 frames (same algorithm, frozen config);
  2. LithoBench printed-vs-design centroid offset of isolated features per set.

  python tools/trace_numbers.py   -> results/<figures>/traceability.json
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

if __name__ == "__main__":
    import numpy as np
    import torch
    from scipy import ndimage

    import registration as rg
    import sem_sim as ss
    from data_lithobench import load_config, load_tile
    cfg = load_config()
    sc, rc, S, M = cfg["sim"], dict(cfg["registration"]), cfg["lithobench"]["crop_px"], cfg["sim"]["margin_px"]
    sp = json.load(open(os.path.join(cfg["paths"]["results"], "splits.json")))
    tiles = [("MetalSet", t) for t in sp["val"]["MetalSet"][10:30]] + [("ViaSet", t) for t in sp["val"]["ViaSet"][10:30]]
    crops = [ss.load_crop(cfg, s, t, *ss.choose_crop(cfg, s, t, S), S) for s, t in tiles]
    a = np.stack([c[0] for c in crops]); T = np.stack([c[1] for c in crops])
    o = ss.simulate(a, ss.sample_params(np.random.default_rng(1), sc, len(tiles)), sc, S, seed=3, frames=[2, 64])
    reg = {}
    for name, TT, cw in (("cad_template", T, rc["corner_weight_px"]), ("printed_template", o["F"] > 0, 0.0)):
        r2 = dict(rc, corner_weight_px=cw)
        tm, wt = rg.design_template(TT, r2)
        for fr in (2, 64):
            A, b = rg.register(torch.as_tensor(o["noisy"][fr], device="cuda")[:, None], torch.as_tensor(tm, device="cuda")[:, None],
                               r2, M, weight=torch.as_tensor(wt, device="cuda")[:, None])
            ec, em = rg.registration_error(A.cpu().numpy(), b.cpu().numpy(), o["A"], o["b"], S)
            reg[f"{name}_{fr}fr"] = {"centre_err_median_px": float(np.median(ec)), "centre_err_p90_px": float(np.percentile(ec, 90)),
                                     "corner_err_median_px": float(np.median(em)), "n_crops": len(tiles)}
    lb = cfg["paths"]["lithobench"]
    off = {}
    for s, ids in (("ViaSet", sp["val"]["ViaSet"][:40]), ("StdContactTest", sp["test"]["StdContactTest"][:20]),
                   ("MetalSet", sp["val"]["MetalSet"][:20]), ("StdMetal", sp["test"]["StdMetal"][:20])):
        dx, dy = [], []
        for t in ids:
            d = load_tile(lb, s, t, ("target", "printed"))
            Tm, P = d["target"] > 0.5, d["printed"] > 0.5
            lab, n = ndimage.label(Tm)
            for k in range(1, n + 1):
                m = lab == k
                pl, pn = ndimage.label(P & ndimage.binary_dilation(m, iterations=12))
                if pn != 1:
                    continue
                cy, cx = ndimage.center_of_mass(m); py, px = ndimage.center_of_mass(pl > 0)
                dx.append(px - cx); dy.append(py - cy)
        off[s] = {"n_features": len(dx), "dx_median_px": float(np.median(dx)), "dy_median_px": float(np.median(dy)),
                  "dx_sd_px": float(np.std(dx)), "dy_sd_px": float(np.std(dy))}
    out = next(os.path.join(ROOT, "results", d) for d in ("figures", "final") if os.path.isdir(os.path.join(ROOT, "results", d)))
    json.dump({"registration_validation_crops": reg, "printed_minus_design_centroid_offset": off,
               "note": "1 px = 1 nm; registration uses the frozen config (coarse translation, matched template)"},
              open(os.path.join(out, "traceability.json"), "w"), indent=1)
    print(json.dumps(reg, indent=0)); print(json.dumps(off, indent=0))
