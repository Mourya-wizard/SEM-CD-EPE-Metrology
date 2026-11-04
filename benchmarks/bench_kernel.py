"""M8: parity and runtime of the C++/CUDA sub-pixel edge refinement kernel (cuda/) vs classical.refine_edges.

  python benchmarks/bench_kernel.py parity                 # short GPU use; writes results/cuda_kernel/parity.csv (+ parity.json)
  python benchmarks/bench_kernel.py bench                  # EXCLUSIVE GPU window only (results/gpu_exclusive.lock held)
  python benchmarks/bench_kernel.py plot                   # results/cuda_kernel/kernel_runtime.png from benchmark.csv

Methods: cpu (C++/OpenMP), cuda_manual (software bilinear), cuda_texture (hardware linear filtering),
torch (vectorised PyTorch: grid_sample for the K profile samples, then argmax/threshold in torch).
Reference: classical.refine_edges in float64 on the same float32 image (cast to float64).
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "./.cache/mpl")
import argparse
import json
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "build"))
MAIN = os.environ.get("SEM_MAIN_REPO", HERE)  # shared results/raw cache + lock
OUT = os.path.join(HERE, "results", "cuda_kernel")
RP = dict(w_in=4.0, w_out=8.0, step=0.05, frac=0.7, min_prom=0.08)  # configs/default.yaml + tuned frac (M3)


def load_ext():
    import edge_refine_ext
    return edge_refine_ext


def P(rp):
    return [rp[k] for k in ("w_in", "w_out", "step", "frac", "min_prom")]


# ----------------------------------------------------------------------------------------------- data
def synthetic_problem(S, N, seed=0, border_frac=0.0):
    """Random rectangles -> material contrast + edge bloom + noise, Gaussian-filtered (sigma 1.5), float32.
    Points: random boundary pixels with sub-pixel jitter, outward unit normals, t0 = bloom peak +- 0.5 px."""
    g = np.random.default_rng(seed)
    m = np.zeros((S, S), np.float32)
    for _ in range(max(4, (S // 64) ** 2 // 2)):
        x0, y0 = g.integers(0, S, 2)
        w, h = g.integers(8, 80, 2)
        m[y0:y0 + h, x0:x0 + w] = 1.0
    mb = ndimage.gaussian_filter(m, 1.0)
    gy, gx = np.gradient(mb)
    mag = np.hypot(gx, gy)
    img = 0.15 + 0.25 * mb + 0.6 * mag / (mag.max() + 1e-12) + 0.03 * g.standard_normal((S, S)).astype(np.float32)
    img = ndimage.gaussian_filter(img, 1.5).astype(np.float32)
    ys, xs = np.nonzero((mag > 0.25 * mag.max()) & (mb > 0.45) & (mb < 0.55))
    k = g.integers(0, len(xs), N)
    pts = np.stack([xs[k], ys[k]], 1).astype(np.float64) + g.uniform(-0.5, 0.5, (N, 2))
    nrm = -np.stack([gx[ys[k], xs[k]], gy[ys[k], xs[k]]], 1).astype(np.float64)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    t0 = g.uniform(-0.5, 0.5, N)
    nb = int(border_frac * N)  # points near/outside the border with random normals: exercises coordinate clamping
    if nb:
        side = g.uniform(-3, 3, (nb, 2)) + np.where(g.random((nb, 2)) < 0.5, 0, S - 1)
        pts[:nb] = np.where(g.random((nb, 1)) < 0.5, side, np.stack([g.uniform(0, S - 1, nb), side[:, 1]], 1))
        a = g.uniform(0, 2 * np.pi, nb)
        nrm[:nb] = np.stack([np.cos(a), np.sin(a)], 1)
    return img, pts.astype(np.float32), nrm.astype(np.float32), t0.astype(np.float32)


def real_problems(n_tiles=6, doses=(2, 8, 64)):
    """Simulated SEM val images (cached, oracle registration), classical sigma/frac per dose and stage-1 peaks
    as t0. Yields (name, img32, pts, nrm, t0, rp)."""
    import classical
    import evaluate as ev
    import measure as ms
    from data_lithobench import load_config
    cfg = load_config()
    sc = cfg["sim"]
    tuned = json.load(open(os.path.join(HERE, "results", "classical_baseline", "classical_tuning.json")))
    cps = ev.classical_params(cfg, tuned)
    nom = ev.nominal_index(cfg)
    for set_name in ("MetalSet", "ViaSet"):
        d0 = os.path.join(MAIN, "results", "raw", "cache", "val", set_name)
        for f in sorted(os.listdir(d0))[:n_tiles]:
            d = ev.load_cache(os.path.join(d0, f))
            q, ns, _ = ms.probes_to_sem(d["probes"], d["A"][nom], d["b"][nom])
            for fr in doses:
                cp = cps[str(fr)]
                noisy = ev.eval_noisy(d["clean"][nom], fr, int(d["seed"]) * 100 + nom, 0, sc)
                img = classical.smooth(noisy, cp["sigma"]).astype(np.float32)
                tp = classical.coarse_peaks(img.astype(np.float64), q, ns, cp)
                ok = np.isfinite(tp)
                rp = {k: cp[k] for k in RP}
                yield (f"{set_name}/{f[:-4]}/dose{fr}", img, q[ok].astype(np.float32), ns[ok].astype(np.float32),
                       tp[ok].astype(np.float32), rp)


def reference(img32, pts, nrm, t0, rp):
    import classical
    return classical.refine_edges(img32.astype(np.float64), pts.astype(np.float64), nrm.astype(np.float64),
                                  t0.astype(np.float64), rp)


# ----------------------------------------------------------------------------------------------- torch
def torch_refine(img, pts, nrm, t0, rp):
    """Same algorithm in PyTorch on device tensors: img (H,W), pts/nrm (N,2), t0 (N,), all float32."""
    import torch
    import torch.nn.functional as F
    H, W = img.shape
    K = int(round((rp["w_in"] + rp["w_out"]) / rp["step"])) + 1
    kin = int(round(2 * rp["w_in"] / rp["step"])) + 1
    ar = torch.arange(K, device=img.device)
    rel = -rp["w_in"] + rp["step"] * ar.float()
    t = t0[:, None] + rel[None]
    gx = (pts[:, 0:1] + nrm[:, 0:1] * t) * (2.0 / (W - 1)) - 1.0
    gy = (pts[:, 1:2] + nrm[:, 1:2] * t) * (2.0 / (H - 1)) - 1.0
    v = F.grid_sample(img[None, None], torch.stack([gx, gy], -1)[None], mode="bilinear",
                      padding_mode="border", align_corners=True)[0, 0]          # (N,K); border == mode="nearest"
    vp, kp = v[:, :kin].max(1)
    base = torch.where(ar[None] >= kp[:, None], v, torch.inf).amin(1)
    level = base + rp["frac"] * (vp - base)
    below = (ar[None] > kp[:, None]) & (v < level[:, None])
    has = below.any(1)
    kc = torch.where(has, below.to(torch.uint8).argmax(1), 1)
    va = v.gather(1, (kc - 1)[:, None])[:, 0]
    vb = v.gather(1, kc[:, None])[:, 0]
    d = va - vb
    tc = rel[kc - 1] + rp["step"] * (va - level) / torch.where(d != 0, d, torch.ones_like(d))
    valid = has & (vp - base >= rp["min_prom"]) & (kp > 0)
    return torch.where(valid, t0 + tc, torch.nan), valid


def run_torch_numpy(img, pts, nrm, t0, rp):
    import torch
    dev = "cuda"
    T = [torch.from_numpy(a).to(dev) for a in (img, pts, nrm, t0)]
    t, v = torch_refine(*T, rp)
    return t.cpu().numpy(), v.cpu().numpy()


def emulate_texture(img, pts, nrm, t0, rp, frac_bits=8):
    """float64 numpy version of the algorithm whose bilinear weights are rounded to `frac_bits` fractional bits,
    i.e. a software model of hardware texture filtering (used only to explain texture-vs-reference deviations)."""
    H, W = img.shape
    K = int(round((rp["w_in"] + rp["w_out"]) / rp["step"])) + 1
    kin = int(round(2 * rp["w_in"] / rp["step"])) + 1
    rel = -rp["w_in"] + rp["step"] * np.arange(K)
    t = t0.astype(np.float64)[:, None] + rel[None]
    x = np.clip(pts[:, 0:1] + nrm[:, 0:1] * t, 0, W - 1)
    y = np.clip(pts[:, 1:2] + nrm[:, 1:2] * t, 0, H - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    q = 2.0 ** frac_bits
    fx, fy = np.round((x - x0) * q) / q, np.round((y - y0) * q) / q
    I = img.astype(np.float64)
    v = (1 - fy) * ((1 - fx) * I[y0, x0] + fx * I[y0, x1]) + fy * ((1 - fx) * I[y1, x0] + fx * I[y1, x1])
    N, r = len(t0), np.arange(len(t0))
    kp = np.argmax(v[:, :kin], 1)
    vp = v[r, kp]
    base = np.where(np.arange(K)[None] >= kp[:, None], v, np.inf).min(1)
    level = base + rp["frac"] * (vp - base)
    below = (np.arange(K)[None] > kp[:, None]) & (v < level[:, None])
    has = below.any(1)
    kc = np.where(has, np.argmax(below, 1), 1)
    va, vb = v[r, kc - 1], v[r, kc]
    tc = rel[kc - 1] + rp["step"] * (va - level) / np.where(va - vb != 0, va - vb, 1.0)
    valid = has & (vp - base >= rp["min_prom"]) & (kp > 0)
    return np.where(valid, t0 + tc, np.nan), valid


# ----------------------------------------------------------------------------------------------- parity
def compare(t_ref, v_ref, t, v):
    v = np.asarray(v).astype(bool)
    both = v_ref & v
    dt = np.abs(t[both].astype(np.float64) - t_ref[both])
    pct = lambda q: float(np.percentile(dt, q)) if len(dt) else np.nan  # noqa: E731
    return dict(n=len(v), n_valid_ref=int(v_ref.sum()), valid_agree=float((v_ref == v).mean()),
                n_valid_mismatch=int((v_ref != v).sum()), n_both_valid=int(both.sum()),
                max_abs_px=float(dt.max()) if len(dt) else np.nan, mean_abs_px=float(dt.mean()) if len(dt) else np.nan,
                p999_abs_px=pct(99.9), p99_abs_px=pct(99), n_gt_0p01px=int((dt > 0.01).sum()),
                n_gt_0p1px=int((dt > 0.1).sum()))


def run_methods(ext, img, pts, nrm, t0, rp, threads):
    out = {"cpu": ext.cpu(img, pts, nrm, t0, *P(rp), threads=threads)[:2]}
    for mode in ("manual", "texture"):
        out["cuda_" + mode] = ext.cuda(img, pts, nrm, t0, *P(rp), mode=mode)[:2]
    out["torch_grid_sample"] = run_torch_numpy(img, pts, nrm, t0, rp)
    return out


def parity(args):
    ext = load_ext()
    os.makedirs(OUT, exist_ok=True)
    groups = {}  # dataset group -> method -> list of (t_ref, v_ref, t, v)
    def add(group, img, pts, nrm, t0, rp):  # noqa: E306
        t_ref, v_ref = reference(img, pts, nrm, t0, rp)
        res = run_methods(ext, img, pts, nrm, t0, rp, args.threads)
        res["emulated_8bit_weights_f64"] = emulate_texture(img, pts, nrm, t0, rp)
        for m, (t, v) in res.items():
            groups.setdefault(group, {}).setdefault(m, []).append((t_ref, v_ref, t, v))
        # texture vs its software model: if ~0, texture deviations are explained by 8-bit weight quantisation
        te, ve = res["emulated_8bit_weights_f64"]
        groups[group].setdefault("cuda_texture_vs_8bit_emulation", []).append((te, ve) + tuple(res["cuda_texture"]))
    for S, N, bf in ((512, 20000, 0.1), (2048, 50000, 0.05)):
        add(f"synthetic_{S}px", *synthetic_problem(S, N, seed=S, border_frac=bf), RP)
        print("synthetic", S, flush=True)
    n_real = 0
    for name, img, pts, nrm, t0, rp in real_problems(args.tiles):
        add("simulated_sem_val", img, pts, nrm, t0, rp)
        n_real += 1
    print("real images", n_real, flush=True)
    rows = []
    for g, ms in groups.items():
        for m, lst in ms.items():
            cat = [np.concatenate([x[i] for x in lst]) for i in range(4)]
            rows.append(dict(dataset=g, method=m, n_images=len(lst), **compare(*cat)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "parity.csv"), index=False)
    print(df.to_string(index=False))


# ----------------------------------------------------------------------------------------------- benchmark
def gpu_processes():
    try:
        return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except Exception as e:  # noqa: BLE001
        return [f"nvidia-smi failed: {e}"]


def check_exclusive():
    lock = os.path.join(MAIN, "results", "gpu_exclusive.lock")
    st = json.load(open(os.path.join(MAIN, "results", "queue_status.json")))
    running = [j["name"] for j in st.get("jobs", []) if j.get("state") == "running"]
    if not os.path.exists(lock) or running:
        sys.exit(f"refusing to benchmark: lock={os.path.exists(lock)} running={running}")
    return dict(lock=lock, queue_running=running, gpu_compute_processes_at_start=gpu_processes())


def wall(f):
    def g():
        s = time.perf_counter()
        f()
        return (time.perf_counter() - s) * 1e3
    return g


def median_ms(fn, reps, warm):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        ts.append(fn())
    return float(np.median(ts)), float(np.percentile(ts, 10)), float(np.percentile(ts, 90))


def bench(args):
    import torch
    ext = load_ext()
    meta = check_exclusive() if not args.no_lock_check else dict(note="lock check skipped (not a valid timing run)")
    out = args.out or OUT
    os.makedirs(out, exist_ok=True)
    thr = args.cpu_threads or os.cpu_count()
    torch.set_num_threads(4)
    rp, reps, warm = RP, args.reps, args.warmup
    rows = []
    def rec(**kw):  # noqa: E306
        rows.append(kw)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in kw.items()}, flush=True)
    for S in args.sizes:
        for N in args.npts:
            img, pts, nrm, t0 = synthetic_problem(S, N, seed=1)
            if args.order == "sorted":
                o = np.lexsort((pts[:, 0] // 32, pts[:, 1] // 32))
                pts, nrm, t0 = pts[o].copy(), nrm[o].copy(), t0[o].copy()
            base = dict(image_px=S, n_points=N, order=args.order)
            # CPU / OpenMP (no transfer: kernel == end-to-end)
            m, lo, hi = median_ms(wall(lambda: ext.cpu(img, pts, nrm, t0, *P(rp), threads=thr)), reps, warm)
            rec(**base, method="cpu_openmp", timing="kernel", median_ms=m, p10_ms=lo, p90_ms=hi, threads=thr)
            rec(**base, method="cpu_openmp", timing="end_to_end", median_ms=m, p10_ms=lo, p90_ms=hi, threads=thr)
            for mode in ("manual", "texture"):
                tex = mode == "texture"
                g = ext.GpuProblem()
                g.set_image(img, tex)
                g.set_points(pts, nrm, t0)
                for blk in ([64, 128, 256] if args.block_sweep else [128]):
                    m, lo, hi = median_ms(lambda: g.run(tex, *P(rp), block=blk), reps, warm)
                    rec(**base, method=f"cuda_{mode}", timing="kernel", median_ms=m, p10_ms=lo, p90_ms=hi, block=blk)
                def reuse():  # noqa: E306  persistent device buffers: H2D image + points, kernel, D2H each call
                    g.set_image(img, tex); g.set_points(pts, nrm, t0); g.run(tex, *P(rp)); g.download()
                m, lo, hi = median_ms(wall(reuse), reps, warm)
                rec(**base, method=f"cuda_{mode}", timing="end_to_end_reuse", median_ms=m, p10_ms=lo, p90_ms=hi, block=128)
                del g
                m, lo, hi = median_ms(wall(lambda: ext.cuda(img, pts, nrm, t0, *P(rp), mode=mode)), reps, warm)
                rec(**base, method=f"cuda_{mode}", timing="end_to_end", median_ms=m, p10_ms=lo, p90_ms=hi, block=128)
                tm = ext.cuda(img, pts, nrm, t0, *P(rp), mode=mode)[2]
                rows[-1].update({k: float(v) for k, v in tm.items() if k != "total_wall_ms"})
            T = [torch.from_numpy(a).cuda() for a in (img, pts, nrm, t0)]
            def tk():  # noqa: E306
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); torch_refine(*T, rp); e1.record(); torch.cuda.synchronize()
                return e0.elapsed_time(e1)
            m, lo, hi = median_ms(tk, reps, warm)
            rec(**base, method="torch_grid_sample", timing="kernel", median_ms=m, p10_ms=lo, p90_ms=hi)
            def te():  # noqa: E306
                t_, v_ = run_torch_numpy(img, pts, nrm, t0, rp)
                torch.cuda.synchronize()
            m, lo, hi = median_ms(wall(te), reps, warm)
            rec(**base, method="torch_grid_sample", timing="end_to_end", median_ms=m, p10_ms=lo, p90_ms=hi)
            rec(**base, method="torch_grid_sample", timing="peak_mem_GB",
                median_ms=float(torch.cuda.max_memory_allocated() / 1e9))
            del T
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    df = pd.DataFrame(rows)
    df["points_per_s"] = np.where(df.timing != "peak_mem_GB", df.n_points / (df.median_ms / 1e3), np.nan)
    tag = "" if args.order == "random" else "_" + args.order
    df.to_csv(os.path.join(out, f"benchmark{tag}.csv"), index=False)
    meta.update(gpu_compute_processes_at_end=gpu_processes(), cpu=platform.processor(), cpu_logical_cores=os.cpu_count(),
                omp_threads=thr, omp_max_threads_default=ext.omp_max_threads(), reps=reps, warmup=warm,
                torch=torch.__version__, gpu=torch.cuda.get_device_name(0), params=rp, order=args.order,
                timing_notes="kernel: CUDA events around the launch (cuda_*), torch.cuda.Event (torch), wall clock (cpu). "
                             "end_to_end: wall clock from host numpy inputs to host numpy outputs incl. cudaMalloc, H2D, "
                             "texture/array setup, kernel, D2H (pageable memory). median of `reps` after `warmup`.",
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(meta, open(os.path.join(out, f"bench_meta{tag}.json"), "w"), indent=1)


def summarize_and_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    o = args.out or OUT
    df = pd.read_csv(os.path.join(o, "benchmark.csv"))
    par = pd.read_csv(os.path.join(OUT, "parity.csv"))
    meta = json.load(open(os.path.join(o, "bench_meta.json")))
    d = df[(df.timing != "peak_mem_GB") & (df.get("block", 128).fillna(128) == 128)]
    from matplotlib.lines import Line2D
    sizes = sorted(d.image_px.unique())
    fig, axs = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4.6), sharey=True, squeeze=False)
    colors = {"cpu_openmp": "#4e79a7", "cuda_manual": "#e15759", "cuda_texture": "#f28e2b", "torch_grid_sample": "#59a14f"}
    styles = {"kernel": "-", "end_to_end_reuse": "--", "end_to_end": ":"}
    for ax, S in zip(axs[0], sizes):
        for m, c in colors.items():
            for tmg, ls in styles.items():
                if m == "torch_grid_sample" and tmg != "kernel":   # torch's caching allocator == reused buffers
                    ls = "--" if tmg == "end_to_end" else None
                if ls is None or (m == "cpu_openmp" and tmg != "kernel"):
                    continue
                s = d[(d.image_px == S) & (d.method == m) & (d.timing == tmg)].sort_values("n_points")
                ax.loglog(s.n_points, s.median_ms, ls, marker="o", ms=4, color=c)
        ax.set_title(f"{S} x {S} px image"); ax.set_xlabel("edge points"); ax.grid(True, which="both", alpha=0.3)
    axs[0, 0].set_ylabel("median time [ms]")
    h = [Line2D([], [], color=c, marker="o", label=m) for m, c in colors.items()]
    h += [Line2D([], [], color="k", ls="-", label="kernel only (CPU: whole call)"),
          Line2D([], [], color="k", ls="--", label="end-to-end, reused device buffers (H2D+kernel+D2H)"),
          Line2D([], [], color="k", ls=":", label="end-to-end incl. cudaMalloc/array setup")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"Sub-pixel edge refinement runtime: {meta['gpu']} vs CPU ({meta['cpu_logical_cores']} logical cores, "
                 f"OpenMP {meta['omp_threads']} threads); median of {meta['reps']} runs", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(o, "kernel_runtime.png"), dpi=130)
    k = d.pivot_table(index=["image_px", "n_points"], columns=["method", "timing"], values="median_ms")
    big = k.loc[(4096, 1000000)] if (4096, 1000000) in k.index else k.iloc[-1]
    summ = dict(
        benchmark_meta=meta,
        largest_case={f"{m}/{t}_ms": float(v) for (m, t), v in big.items()},
        speedup_vs_cpu_kernel={f"{S}px_{N}": {m: float(k.loc[(S, N)][("cpu_openmp", "kernel")] / k.loc[(S, N)][(m, "kernel")])
                                              for m in ("cuda_manual", "cuda_texture", "torch_grid_sample")} for S, N in k.index},
        speedup_vs_cpu_end_to_end={f"{S}px_{N}": {m: float(k.loc[(S, N)][("cpu_openmp", "end_to_end")] / k.loc[(S, N)][(m, "end_to_end")])
                                                  for m in ("cuda_manual", "cuda_texture", "torch_grid_sample")} for S, N in k.index},
        speedup_vs_cpu_end_to_end_reuse={f"{S}px_{N}": {m: float(k.loc[(S, N)][("cpu_openmp", "end_to_end")] / k.loc[(S, N)][(m, "end_to_end_reuse")])
                                                        for m in ("cuda_manual", "cuda_texture")} for S, N in k.index},
        max_points_per_s={m: float(d[(d.method == m) & (d.timing == "kernel")].points_per_s.max()) for m in colors},
        parity=par.to_dict(orient="records"))
    so = os.path.join(o, "benchmark_sorted.csv")
    if os.path.exists(so):  # spatially sorted vs random point order (cache-locality probe, kernel only)
        b = pd.read_csv(so)
        b = b[(b.timing == "kernel") & (b.get("block", 128).fillna(128) == 128)]
        r = d[d.timing == "kernel"].merge(b, on=["image_px", "n_points", "method"], suffixes=("_random", "_sorted"))
        summ["kernel_ms_random_vs_sorted"] = r[["image_px", "n_points", "method", "median_ms_random", "median_ms_sorted"]].to_dict(orient="records")
    json.dump(summ, open(os.path.join(o, "summary.json"), "w"), indent=1)
    print(json.dumps({k_: v for k_, v in summ.items() if k_ != "parity"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["parity", "bench", "plot"])
    ap.add_argument("--threads", type=int, default=4, help="OpenMP threads for parity runs")
    ap.add_argument("--tiles", type=int, default=30, help="val tiles per set for parity")
    ap.add_argument("--cpu-threads", type=int, default=0, help="OpenMP threads for the benchmark (0 = all logical cores)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 2048, 4096])
    ap.add_argument("--npts", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--order", choices=["random", "sorted"], default="random")
    ap.add_argument("--block-sweep", action="store_true")
    ap.add_argument("--out", default=None, help="benchmark output dir (default results/cuda_kernel)")
    ap.add_argument("--no-lock-check", action="store_true", help="dry run only; results are not valid timings")
    a = ap.parse_args()
    {"parity": parity, "bench": bench, "plot": summarize_and_plot}[a.cmd](a)
