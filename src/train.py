"""One training run from one config (launched by tools/run_queue.py as its own process).

  python src/train.py --build-bank                 # once: precompute training crops (CPU, <= max_cpu_workers)
  python src/train.py --run configs/runs/<name>.yaml [--pilot N]

Training data are generated on the GPU on the fly from the crop bank: random dihedral augmentation, random
imaging parameters in the training ranges, exposure-like threshold, local bumps, roughness, random dose, noise,
then the SAME registration algorithm as evaluation (CAD template) to warp the CAD input with the ESTIMATED
registration. Target = TSDF in the SEM frame from the truth field warped with the TRUE registration.
Checkpoint selection: frozen metric = mean validation TMU-style CD (cd + contact gauges, learned-only zero
crossing) at eval.selection_dose frames, rep 0, inside-range conditions; validation failure rate logged beside it.
"""
import os

os.environ.setdefault("HF_HOME", "./.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "./.cache/torch")
os.environ.setdefault("XDG_CACHE_HOME", "./.cache")
os.environ.setdefault("MPLCONFIGDIR", "./.cache/mpl")

import argparse
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registration as rg  # noqa: E402
import sem_sim as ss  # noqa: E402
from data_lithobench import load_config  # noqa: E402
from models import build_model, n_params  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------------------------- crop bank
def _bank_item(args):
    cfg, i, set_name, tile, seed = args
    from data_lithobench import load_tile
    from scipy import ndimage
    rng = np.random.default_rng(seed)
    S, M, lc = cfg["lithobench"]["crop_px"], cfg["sim"]["margin_px"], cfg["lithobench"]
    lo, hi = lc["active_lo_px"], lc["active_hi_px"] - S
    if set_name == "ViaSet":                    # sparse: centre the FOV near a random via
        t = load_tile(cfg["paths"]["lithobench"], set_name, tile, ("target",))["target"] > 0.5
        lab, n = ndimage.label(t)
        if n:
            cy, cx = ndimage.center_of_mass(lab == rng.integers(1, n + 1))
            y0 = int(np.clip(cy - rng.uniform(64, S - 64), lo, hi)); x0 = int(np.clip(cx - rng.uniform(64, S - 64), lo, hi))
        else:
            y0, x0 = int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1))
    else:
        y0, x0 = int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1))
    a, T = ss.load_crop(cfg, set_name, tile, y0, x0, S)
    tm, wt = rg.design_template(T[None], cfg["registration"])
    return i, a.astype(np.float16), T.astype(np.uint8), tm[0].astype(np.float16), wt[0].astype(np.float16), (set_name, tile, y0, x0)


def build_bank(cfg):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    tc = cfg["train"]
    bank = os.path.join(cfg["paths"]["data_root_abs"], "bank")
    os.makedirs(bank, exist_ok=True)
    sp = json.load(open(os.path.join(cfg["paths"]["results"], "splits.json")))
    rng = np.random.default_rng(tc["bank_seed"])
    items = []
    for s in ("MetalSet", "ViaSet"):
        ids = rng.choice(sp["train"][s], tc["bank_per_set"], replace=False)
        items += [(s, t) for t in ids]
    N, H = len(items), cfg["lithobench"]["crop_px"] + 2 * cfg["sim"]["margin_px"]
    arr = {k: np.lib.format.open_memmap(os.path.join(bank, f"{k}.npy"), "w+", dt, (N, H, H))
           for k, dt in (("aerial", np.float16), ("design", np.uint8), ("tmpl", np.float16), ("weight", np.float16))}
    meta = [None] * N
    args = [(cfg, i, s, t, int(tc["bank_seed"]) * 100000 + i) for i, (s, t) in enumerate(items)]
    t0 = time.time()
    with ProcessPoolExecutor(cfg["eval"]["max_cpu_workers"], mp_context=get_context("spawn")) as ex:
        for k, (i, a, T, tm, wt, mt) in enumerate(ex.map(_bank_item, args, chunksize=8)):
            arr["aerial"][i], arr["design"][i], arr["tmpl"][i], arr["weight"][i], meta[i] = a, T, tm, wt, mt
            if (k + 1) % 500 == 0:
                print(f"bank {k + 1}/{N} {time.time() - t0:.0f}s", flush=True)
    for v in arr.values():
        v.flush()
    json.dump({"items": meta, "n": N}, open(os.path.join(bank, "meta.json"), "w"))
    print(f"bank done {N} crops {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------------------------- batches
class Bank:
    def __init__(self, cfg):
        d = os.path.join(cfg["paths"]["data_root_abs"], "bank")
        self.a = {k: np.load(os.path.join(d, f"{k}.npy"), mmap_mode="r") for k in ("aerial", "design", "tmpl", "weight")}
        self.n = self.a["aerial"].shape[0]

    def get(self, idx, device):
        idx = np.sort(idx)
        return [torch.as_tensor(np.ascontiguousarray(self.a[k][idx]), device=device).float()[:, None]
                for k in ("aerial", "design", "tmpl", "weight")]


def _dihedral(ts, k, flip):
    out = [torch.rot90(t, k, (-2, -1)) for t in ts]
    return [torch.flip(t, (-1,)) for t in out] if flip else out


def make_batch(cfg, bank, rng, gen, device, cad=True, frames=None):
    """Returns inputs (B,C,S,S), target TSDF (B,1,S,S), weight (B,1,S,S)."""
    sc, rc, tc = cfg["sim"], cfg["registration"], cfg["train"]
    S, M, B = cfg["lithobench"]["crop_px"], sc["margin_px"], tc["batch"]
    a, T, tm, wt = bank.get(rng.choice(bank.n, B, replace=False), device)
    # same dihedral transform for the whole batch member set (per-sample would need a loop; per batch is enough)
    a, T, tm, wt = _dihedral([a, T, tm, wt], int(rng.integers(4)), bool(rng.integers(2)))
    p = ss.to_torch(ss.sample_params(rng, sc, B), device)
    F, _ = ss.build_truth(a, p, sc, gen, bumps=True)
    clean, A, b = ss.render_clean(F, p, sc, S)
    fr = torch.as_tensor(rng.choice(sc["frames"], B) if frames is None else np.full(B, frames), device=device,
                         dtype=torch.float32)[:, None, None, None]
    lam = clean.clamp(min=0) * fr * sc["electrons_per_frame"]
    noisy = torch.poisson(lam, generator=gen) / (fr * sc["electrons_per_frame"])
    noisy = noisy + sc["detector_sigma"] * torch.randn(noisy.shape, generator=gen, device=device)
    x = (noisy - noisy.mean((-1, -2), keepdim=True)) / noisy.std((-1, -2), keepdim=True).clamp(min=1e-6)
    if cad:
        with torch.no_grad():
            Ae, be = rg.register(noisy, tm, rc, M, weight=wt)
        x = torch.cat([x, rg.warp_design_to_sem(T, Ae, be, S)], 1)
    d = ss.signed_distance(F, sc)
    trunc = cfg["model"]["tsdf_trunc_px"]
    tgt = (rg._warp(d, A, b, S) / trunc).clamp(-1, 1)
    w = 1.0 + tc["edge_weight"] * (tgt.abs() < 1).float()
    return x, tgt, w


# ----------------------------------------------------------------------------------------------- validation
def predict_tsdf(model, imgs, cads, device, bs=8):
    """imgs (N,S,S) noisy SEM, cads (N,S,S) or None. Returns TSDF in px (N,S,S) np."""
    out = []
    model.eval()
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        for i in range(0, len(imgs), bs):
            x = torch.as_tensor(np.stack(imgs[i:i + bs]), device=device, dtype=torch.float32)[:, None]
            x = (x - x.mean((-1, -2), keepdim=True)) / x.std((-1, -2), keepdim=True).clamp(min=1e-6)
            if cads is not None:
                x = torch.cat([x, torch.as_tensor(np.stack(cads[i:i + bs]), device=device, dtype=torch.float32)[:, None]], 1)
            out.append(model(x).float().cpu().numpy()[:, 0])
    return np.concatenate(out)


def cad_in_sem(T, A, b, S, device):
    return rg.warp_design_to_sem(torch.as_tensor(T, device=device, dtype=torch.float32)[None, None],
                                 torch.as_tensor(A, device=device, dtype=torch.float32)[None],
                                 torch.as_tensor(b, device=device, dtype=torch.float32)[None], S)[0, 0].cpu().numpy()


def validate(model, cfg, run, device, n_per_set, split="val"):
    """Frozen selection metric on cached val tiles (see module docstring)."""
    import evaluate as ev
    import gauges as gg
    import measure as ms
    import pandas as pd
    sc, S = cfg["sim"], cfg["lithobench"]["crop_px"]
    dose = cfg["eval"]["selection_dose"]
    inside = [i for i, c in enumerate(ev.conditions(cfg)) if c["inside"]]
    tiles = [(s, t) for s in ("MetalSet", "ViaSet") for t in [t for s2, t in ev.eval_tiles(cfg, split) if s2 == s][:n_per_set]]
    rows = []
    trunc = cfg["model"]["tsdf_trunc_px"]
    for s, t in tiles:
        d = ev.load_cache(os.path.join(cfg["paths"]["results"], "raw", "cache", split, s, t + ".npz"))
        r = np.load(os.path.join(cfg["paths"]["results"], "raw", "reg", split, s, t + ".npz"))
        jl = [tuple(j) for j in r["jobs"].tolist()]
        imgs, cads, regs = [], [], []
        for c in inside:
            j = jl.index((c, dose, 0))
            imgs.append(ev.eval_noisy(d["clean"][c], dose, int(d["seed"]) * 100 + c, 0, sc))
            regs.append((r["A"][j], r["b"][j]))
            if run["cad"]:
                cads.append(cad_in_sem(d["T"], r["A"][j], r["b"][j], S, device))
        tsdf = predict_tsdf(model, imgs, cads if run["cad"] else None, device) * trunc
        for k, c in enumerate(inside):
            res = ms.measure_learned(tsdf[k], imgs[k], d["probes"], regs[k][0], regs[k][1], cfg["gauges"]["probe_half_window_px"])
            gv = gg.gauge_values(d["probes"], *res["learned"])
            rows.append(gv.assign(cond=c, dose=dose, rep=0, set=s, tile=t))
    meas = pd.concat(rows, ignore_index=True)
    m = ev.merged(cfg, meas, ev.load_truth(cfg, split, tiles))
    a_cd, a_ct = ev.accuracy_regression(m[m.kind == "cd"]), ev.accuracy_regression(m[m.kind == "contact"])
    model.train()
    return dict(sel_metric=float(np.nanmean([a_cd["tmu3s"], a_ct["tmu3s"]])), cd_tmu3s=a_cd["tmu3s"],
                contact_tmu3s=a_ct["tmu3s"], val_fail=float(m[m.kind.isin(["cd", "contact"])].fail.mean()),
                cd_slope=a_cd["slope"], cd_bias=a_cd["bias"])


# ----------------------------------------------------------------------------------------------- run
def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def train(run_path, pilot_iters=None):
    cfg = load_config()
    run = yaml.safe_load(open(run_path))
    tc = dict(cfg["train"]); tc.update(run.get("train", {}))
    cfg["train"] = tc
    run_dir = os.path.join(ROOT, "checkpoints", run["name"] + ("_pilot" if pilot_iters else ""))
    os.makedirs(run_dir, exist_ok=True)
    status_p, log_p = os.path.join(run_dir, "status.json"), os.path.join(run_dir, "log.jsonl")
    device = "cuda"
    mf = run.get("mem_fraction", cfg["queue"]["mem_fraction"])     # safe per-run allocation from the M4 pilot
    torch.cuda.set_per_process_memory_fraction(float(mf))
    torch.manual_seed(run["seed"]); np.random.seed(run["seed"])
    model = build_model(cfg["model"], run["op"], 2 if run["cad"] else 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    iters = pilot_iters or tc["iters"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / tc["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(i, iters) / iters)))
    it0, best = 0, {"sel_metric": float("inf")}
    last_p = os.path.join(run_dir, "last.pt")
    if os.path.exists(last_p) and not pilot_iters:                 # resume
        ck = torch.load(last_p, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        it0, best = ck["iter"], ck["best"]
    rng = np.random.default_rng(run["seed"] * 1000 + it0)
    gen = torch.Generator(device=device).manual_seed(run["seed"] * 1000 + it0)
    bank = Bank(cfg)
    info = dict(name=run["name"], op=run["op"], cad=run["cad"], seed=run["seed"], params=n_params(model),
                width=cfg["model"]["width"][run["op"]])
    _atomic_json(status_p, dict(info, state="running", iter=it0, iters=iters, best=best, pid=os.getpid()))
    t_last, n_img, t_data, t_step = time.time(), 0, 0.0, 0.0
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for it in range(it0, iters):
        ta = time.time()
        x, tgt, w = make_batch(cfg, bank, rng, gen, device, cad=run["cad"])
        torch.cuda.synchronize(); tb = time.time()
        with torch.autocast(device, dtype=torch.bfloat16):
            pred = model(x)
        loss = (w * (pred.float() - tgt).abs()).sum() / w.sum()
        if not torch.isfinite(loss):
            _atomic_json(status_p, dict(info, state="nan", iter=it, iters=iters, best=best))
            print("NaN loss at iter", it, flush=True)
            sys.exit(3)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step(); sched.step()
        torch.cuda.synchronize()
        t_data += tb - ta; t_step += time.time() - tb; n_img += x.shape[0]
        if (it + 1) % tc["log_every"] == 0 or it + 1 == iters:
            dt = time.time() - t_last
            rec = dict(iter=it + 1, loss=float(loss), lr=sched.get_last_lr()[0], img_per_s=n_img / dt,
                       data_frac=t_data / max(t_data + t_step, 1e-9), peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
            with open(log_p, "a") as f:
                f.write(json.dumps(rec) + "\n")
            t_last, n_img, t_data, t_step = time.time(), 0, 0.0, 0.0
        if not pilot_iters and ((it + 1) % tc["val_every"] == 0 or it + 1 == iters):
            v = validate(model, cfg, run, device, tc["val_tiles_per_set"], tc.get("val_split", "val"))
            v["iter"] = it + 1
            with open(os.path.join(run_dir, "val.jsonl"), "a") as f:
                f.write(json.dumps(v) + "\n")
            if v["sel_metric"] < best["sel_metric"]:
                best = v
                torch.save({"model": model.state_dict(), "run": run, "iter": it + 1, "val": v, "info": info},
                           os.path.join(run_dir, "best.pt.tmp"))
                os.replace(os.path.join(run_dir, "best.pt.tmp"), os.path.join(run_dir, "best.pt"))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "iter": it + 1, "best": best}, last_p + ".tmp")
            os.replace(last_p + ".tmp", last_p)
            _atomic_json(status_p, dict(info, state="running", iter=it + 1, iters=iters, best=best, pid=os.getpid()))
    _atomic_json(status_p, dict(info, state="done", iter=iters, iters=iters, best=best,
                                peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-bank", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--pilot", type=int, default=None)
    a = ap.parse_args()
    if a.build_bank:
        build_bank(load_config())
    else:
        train(a.run, a.pilot)
