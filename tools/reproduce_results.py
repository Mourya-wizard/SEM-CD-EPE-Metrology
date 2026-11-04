"""Post-training stage of `run_all.sh`: evaluations, real-SEM pipeline, trust analysis, tables and figures.

Every step skips work whose output already exists, so an interrupted run can simply be restarted.
GPU jobs go through tools/run_queue.py; CPU work respects configs/default.yaml eval.max_cpu_workers.

  python tools/reproduce_results.py
"""
import glob
import json
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
sys.path.insert(0, os.path.join(ROOT, "src"))


def run(*args):
    print("[reproduce]", " ".join(str(a) for a in args), flush=True)
    subprocess.run([PY, *[str(a) for a in args]], cwd=ROOT, check=True)


def queue(paths, concurrency=1):
    if paths:
        run("tools/run_queue.py", "start", "--concurrency", concurrency, "--runs", *paths)


def main():
    import evaluate as ev
    from data_lithobench import load_config
    cfg = load_config()
    meas = lambda split, name: os.path.exists(os.path.join(cfg["paths"]["results"], "raw", "meas", split, name + ".pkl.gz"))  # noqa: E731
    jobs_dir = os.path.join(ROOT, "configs", "queue_all")
    os.makedirs(jobs_dir, exist_ok=True)

    # 1. learned-model evaluation jobs (validation + frozen test; oracle registration for the EPE decomposition)
    runs = sorted(os.path.basename(p)[:-5] for pat in ("screen_*", "confirm_*", "cadtest_*")
                  for p in glob.glob(os.path.join(ROOT, "configs", "runs", pat + ".yaml")))
    todo = []
    for r in runs:
        for split, reg in (("val", "est"), ("test", "est")) + ((("test", "oracle"),) if r in ("screen_conv_cad_s1", "screen_dcnv3_cad_s1") else ()):
            tag = f"{r}{'' if reg == 'est' else '_oraclereg'}__learned"
            if not meas(split, tag):
                p = os.path.join(jobs_dir, f"eval_{r}_{split}_{reg}.yaml")
                yaml.safe_dump(dict(name=f"eval_{r}_{split}_{reg}", kind="eval", ckpt=r, split=split, reg=reg), open(p, "w"))
                todo.append(p)
    queue(todo, concurrency=2)                      # evaluation jobs are CPU-bound; two share the GPU well

    # 2. classical frozen test evaluation (CPU)
    tuned = json.load(open(os.path.join(cfg["paths"]["results"], "classical_baseline", "classical_tuning.json")))
    cp = ev.classical_params(cfg, tuned)
    for reg in ("est", "oracle"):
        if not meas("test", f"classical_{reg}"):
            ev.save_meas(ev.run_classical(cfg, "test", cp, reg, label="test"), cfg, "test", f"classical_{reg}")

    # 3. validation screening table (model selection)
    out = os.path.join(cfg["paths"]["results"], "model_eval")
    os.makedirs(out, exist_ok=True)
    names = [f"screen_{o}_cad_s1__{k}" for k in ("learned", "refine") for o in ("conv", "dcnv2", "dcnv3", "dcnv4")] + ["classical_est"]
    if not meas("val", "classical_est"):
        ev.save_meas(ev.run_classical(cfg, "val", cp, "est", label="val"), cfg, "val", "classical_est")
    ev.screening_table(cfg, names, "val").to_csv(os.path.join(out, "val_screening.csv"), index=False)

    # 4. trust analysis on the three DCNv3 seeds
    run("src/confidence.py", "--runs", "screen_dcnv3_cad_s1", "confirm_dcnv3_cad_s2", "confirm_dcnv3_cad_s3",
        "--method", "learned", "--split", "test", "--tag", "dcnv3_3seeds")

    # 5. real SEM (Carinthia-S): split, training through the queue, aggregation, noise
    if not os.path.exists(os.path.join(cfg["paths"]["results"], "carinthia", "carinthia_split.json")):
        run("src/carinthia.py", "split")
    cj = []
    for p in sorted(glob.glob(os.path.join(ROOT, "configs", "runs", "carinthia", "carinthia_*_s*.yaml"))):
        name = os.path.basename(p)[:-5]
        if not os.path.exists(os.path.join(ROOT, "checkpoints", name, "metrics.json")):
            q = os.path.join(jobs_dir, f"{name}.yaml")
            yaml.safe_dump(dict(name=name, kind="other", cmd=["src/carinthia.py", "train", "--run", p], cwd=ROOT), open(q, "w"))
            cj.append(q)
    queue(cj, concurrency=1)
    run("src/carinthia.py", "aggregate")
    run("src/carinthia.py", "noise")

    # 6. learned-model roughness (line/space set), traceability diagnostics, tables and figures
    for r in ("screen_dcnv3_cad_s1", "screen_conv_cad_s1"):
        for sg in (1, 2):
            run("tools/roughness_learned.py", r, sg)
    run("tools/trace_numbers.py")
    run("tools/results_snapshot.py")
    run("src/plots.py", "figures")
    run("tools/plot_lcdu.py")
    run("tools/make_dashboard.py")
    run("tools/make_banner.py")
    print("[reproduce] done: results/ and docs/RESULTS.md are up to date", flush=True)


if __name__ == "__main__":
    main()
