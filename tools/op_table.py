"""Operator source table (M4): implementation source, params, peak training memory, training throughput
(from a short real pilot run per variant, N=1) and inference time (batch 1 and 8, 512x512, bf16 autocast,
median of 50 timed runs after 10 warm-up runs, torch.cuda.synchronize). -> results/operators/operator_sources.csv
Runtime comparisons are only meaningful between implementations of the same kind (all four here are
PyTorch-level ops; none is a custom CUDA kernel).
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from models import build_model, n_params  # noqa: E402
from operators import SOURCES  # noqa: E402

PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join(ROOT, ".venv", "bin", "python")


def pilot(op, iters=120):
    name = f"optable_{op}"
    rp = os.path.join(ROOT, "configs", "runs", "pilot", name + ".yaml")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    yaml.safe_dump(dict(name=name, op=op, cad=True, seed=7), open(rp, "w"))
    d = os.path.join(ROOT, "checkpoints", name + "_pilot")
    if os.path.exists(os.path.join(d, "log.jsonl")):
        os.remove(os.path.join(d, "log.jsonl"))
    subprocess.run([PY, os.path.join(ROOT, "src", "train.py"), "--run", rp, "--pilot", str(iters)], check=True,
                   cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
    recs = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))]
    st = recs[len(recs) // 3:]
    return float(np.mean([r["img_per_s"] for r in st])), max(r["peak_mem_gb"] for r in recs)


def infer_ms(op, bs):
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml")))
    m = build_model(cfg["model"], op, 2).cuda().eval()
    x = torch.randn(bs, 2, 512, 512, device="cuda")
    ts = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(60):
            torch.cuda.synchronize(); t = time.perf_counter()
            m(x)
            torch.cuda.synchronize()
            if i >= 10:
                ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), n_params(m)


if __name__ == "__main__":
    rows = []
    for op in ("conv", "dcnv2", "dcnv3", "dcnv4"):
        ips, mem = pilot(op)
        i1, npar = infer_ms(op, 1)
        i8, _ = infer_ms(op, 8)
        rows.append(dict(op=op, source=SOURCES[op], params=npar, train_img_per_s_N1=round(ips, 2), train_peak_alloc_gb=round(mem, 2),
                         infer_ms_bs1=round(i1, 2), infer_ms_bs8=round(i8, 2)))
        print(rows[-1], flush=True)
    os.makedirs(os.path.join(ROOT, "results", "operators"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "operators", "operator_sources.csv"), index=False)
