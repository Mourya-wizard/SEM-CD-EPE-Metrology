"""Concurrency pilot: run N identical short REAL training workloads at once (N = 1, 2, 4, 6, 8 where memory
allows) and measure per-run and aggregate images/s, peak VRAM, GPU util, CPU util and data-wait fraction.
Choice rule: smallest N beyond which aggregate throughput rises < 10%, with total VRAM < 85%.
NVIDIA MPS is Linux-only (not testable on this Windows/WDDM machine); MIG is not used.

  python tools/gpu_pilot.py [--op dcnv3] [--iters 150] [--ns 1 2 4 6 8]  -> results/gpu_pilot.csv
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import psutil
import pynvml
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join(ROOT, ".venv", "bin", "python")


def sample_util(stop, out):
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    psutil.cpu_percent(None)
    while not stop.is_set():
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        out.append((u.gpu, m.used / 1e9, m.total / 1e9, psutil.cpu_percent(None)))
        time.sleep(1.0)


def run_n(op, n, iters):
    os.makedirs(os.path.join(ROOT, "configs", "runs", "pilot"), exist_ok=True)
    procs, names = [], []
    for k in range(n):
        name = f"pilot_{op}_n{n}_{k}"
        rp = os.path.join(ROOT, "configs", "runs", "pilot", name + ".yaml")
        yaml.safe_dump(dict(name=name, op=op, cad=True, seed=100 + k), open(rp, "w"))
        d = os.path.join(ROOT, "checkpoints", name + "_pilot")
        if os.path.exists(os.path.join(d, "log.jsonl")):
            os.remove(os.path.join(d, "log.jsonl"))
        os.makedirs(d, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        procs.append(subprocess.Popen([PY, "-u", os.path.join(ROOT, "src", "train.py"), "--run", rp, "--pilot", str(iters)],
                                      stdout=open(os.path.join(d, "stdout.log"), "w"), stderr=subprocess.STDOUT, env=env, cwd=ROOT))
        names.append(name)
    stop, samples = threading.Event(), []
    th = threading.Thread(target=sample_util, args=(stop, samples)); th.start()
    rcs = [p.wait() for p in procs]
    stop.set(); th.join()
    per = []
    for name in names:
        recs = [json.loads(l) for l in open(os.path.join(ROOT, "checkpoints", name + "_pilot", "log.jsonl"))]
        steady = recs[len(recs) // 3:]                     # drop warm-up third
        per.append(dict(ips=sum(r["img_per_s"] for r in steady) / len(steady),
                        data=sum(r["data_frac"] for r in steady) / len(steady), mem=max(r["peak_mem_gb"] for r in recs)))
    s = samples[len(samples) // 3:] or samples
    return dict(op=op, n=n, iters=iters, returncodes=str(rcs), per_run_img_s=sum(p["ips"] for p in per) / n,
                aggregate_img_s=sum(p["ips"] for p in per), peak_alloc_gb_per_run=max(p["mem"] for p in per),
                data_wait_frac=sum(p["data"] for p in per) / n, gpu_util_mean=sum(x[0] for x in s) / len(s),
                vram_used_gb_max=max(x[1] for x in samples), vram_total_gb=samples[0][2],
                cpu_util_mean=sum(x[3] for x in s) / len(s))


if __name__ == "__main__":
    import csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="dcnv3")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--ns", type=int, nargs="*", default=[1, 2, 4, 6, 8])
    a = ap.parse_args()
    rows = []
    out = os.path.join(ROOT, "results", "operators", "gpu_pilot.csv")
    for n in a.ns:
        if rows and rows[-1]["vram_used_gb_max"] / rows[-1]["vram_total_gb"] * n / rows[-1]["n"] > 0.85:
            print(f"skip N={n}: projected VRAM > 85%"); continue
        r = run_n(a.op, n, a.iters)
        rows.append(r)
        print(json.dumps(r), flush=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sys.exit(0)
