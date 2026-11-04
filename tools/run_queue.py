"""Local GPU job queue. The ONLY scheduler allowed to start GPU training/eval jobs.

  python tools/run_queue.py start --runs configs/runs/a.yaml ... [--concurrency N] [--keep-alive]
  New jobs: drop a run yaml into configs/queue/ (picked up while the queue runs; with --keep-alive it waits).
  Generic job yaml: {name, kind: train|eval|other, cmd: [script, args...], cwd: <dir>, env: {K: V}} runs
  `<main venv python> -u script args...` in cwd (e.g. Carinthia training from another checkout).
  python tools/run_queue.py status            # print results/queue_status.json
  python tools/run_queue.py stop              # no new launches; running jobs finish (clean stop)
  python tools/run_queue.py kill              # terminate jobs started by this queue (stop flag + terminate)

Each job = one separate Python process (`src/train.py --run <yaml>` or, for kind: eval, `src/evaluate.py
run-learned --run <yaml>`), one config, one run dir (checkpoints/<name>/), one seed. Resume: jobs whose
checkpoints/<name>/status.json says done are skipped; crashed jobs are retried ONCE (train.py resumes from
last.pt). NaN (exit code 3) is not retried. New launches pause while results/gpu_exclusive.lock exists.
Status JSON is written atomically. Concurrency default comes from configs/default.yaml queue.concurrency.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join(ROOT, ".venv", "bin", "python")
STATUS = os.path.join(ROOT, "results", "queue_status.json")
STOP = os.path.join(ROOT, "results", "queue.stop")
LOCK = os.path.join(ROOT, "results", "gpu_exclusive.lock")


def _atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _run_state(name):
    p = os.path.join(ROOT, "checkpoints", name, "status.json")
    try:
        return json.load(open(p)).get("state")
    except Exception:
        return None


def _cmd(run_path, run):
    if run.get("cmd"):                       # generic job: [script, args...] run with the main venv python
        return [PY, "-u"] + [str(c) for c in run["cmd"]]
    if run.get("kind", "train") == "eval":
        return [PY, "-u", os.path.join(ROOT, "src", "evaluate.py"), "run-learned", "--run", run_path]
    return [PY, "-u", os.path.join(ROOT, "src", "train.py"), "--run", run_path]


def _external_jobs(own_pids):
    """Project GPU jobs running that this queue did not start (e.g. from a previous queue process).
    Counts one per job: the Windows venv launcher and its interpreter child share a command line."""
    import psutil
    pat = ("run-learned", "train.py", "carinthia.py")
    procs = {}
    for p in psutil.process_iter(["pid", "ppid", "cmdline"]):
        c = " ".join(p.info["cmdline"] or [])
        if ROOT.replace("\\", "/").split("/")[-1] in c.replace("\\", "/") or "src/" in c:
            if any(k in c for k in pat) and "run_queue" not in c:
                procs[p.info["pid"]] = p.info["ppid"]
    own = set(own_pids)
    for pid in list(own):
        try:
            own |= {c.pid for c in psutil.Process(pid).children(recursive=True)}
        except Exception:
            pass
    return sum(1 for pid, pp in procs.items() if pid not in own and pp not in procs)


def start(runs, concurrency, keep_alive=False):
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    if os.path.exists(STOP):
        os.remove(STOP)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               HF_HOME=os.path.join(ROOT, ".cache", "huggingface"), TORCH_HOME=os.path.join(ROOT, ".cache", "torch"),
               XDG_CACHE_HOME=os.path.join(ROOT, ".cache"), MPLCONFIGDIR=os.path.join(ROOT, ".cache", "mpl"))
    jobs = []
    for rp in runs:
        run = yaml.safe_load(open(rp))
        tag = run["name"] + (":eval" if run.get("kind") == "eval" else "")
        jobs.append(dict(path=os.path.abspath(rp), name=run["name"], tag=tag, kind=run.get("kind", "train"),
                         state="pending", attempts=0, run=run))
    running = {}
    t0 = time.time()
    inbox = os.path.join(ROOT, "configs", "queue")
    os.makedirs(inbox, exist_ok=True)
    while True:
        known = {j["path"] for j in jobs}                        # inbox: jobs added while the queue runs
        for fn in sorted(os.listdir(inbox)):
            rp = os.path.abspath(os.path.join(inbox, fn))
            if fn.endswith(".yaml") and rp not in known:
                run = yaml.safe_load(open(rp))
                tag = run["name"] + (":eval" if run.get("kind") == "eval" else "")
                jobs.append(dict(path=rp, name=run["name"], tag=tag, kind=run.get("kind", "train"), state="pending", attempts=0, run=run))
        for j in jobs:                                           # reap finished processes
            p = running.get(j["tag"])
            if p is None or p.poll() is None:
                continue
            rc = p.returncode
            del running[j["tag"]]
            ok = rc == 0 and (j["kind"] != "train" or j["run"].get("cmd") or _run_state(j["name"]) == "done")
            if ok:
                j["state"] = "done"
            elif rc == 3:
                j["state"] = "nan"
            elif j["attempts"] < 2:
                j["state"] = "pending"                           # retry once (resumes from last.pt)
            else:
                j["state"] = "failed"
            j["returncode"] = rc
        stop = os.path.exists(STOP)
        for j in jobs:                                           # launch
            if stop or os.path.exists(LOCK) or len(running) + _external_jobs([p.pid for p in running.values()]) >= concurrency:
                break
            if j["state"] != "pending":
                continue
            if j["kind"] == "train" and _run_state(j["name"]) == "done":
                j["state"] = "done"
                continue
            mf = j["run"].get("mem_fraction")
            d = os.path.join(ROOT, "checkpoints", j["name"])
            os.makedirs(d, exist_ok=True)
            log = open(os.path.join(d, f"stdout_{j['kind']}_{j['attempts']}.log"), "w")
            running[j["tag"]] = subprocess.Popen(_cmd(j["path"], j["run"]), stdout=log, stderr=subprocess.STDOUT,
                                                 env=dict(env, **{k: str(v) for k, v in j["run"].get("env", {}).items()}),
                                                 cwd=j["run"].get("cwd", ROOT))
            j.update(state="running", attempts=j["attempts"] + 1, pid=running[j["tag"]].pid, started=time.time(), mem_fraction=mf)
        _atomic(STATUS, dict(updated=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_s=round(time.time() - t0),
                             concurrency=concurrency, stop_flag=stop, exclusive_lock=os.path.exists(LOCK),
                             jobs=[{k: v for k, v in j.items() if k != "run"} | {"train_status": _run_state(j["name"])} for j in jobs]))
        if not running and (stop or (all(j["state"] != "pending" for j in jobs) and not keep_alive)):
            break
        time.sleep(15)


def kill():
    open(STOP, "w").close()
    try:
        st = json.load(open(STATUS))
    except Exception:
        return
    import psutil
    for j in st["jobs"]:
        if j.get("state") == "running" and j.get("pid"):
            try:                                   # Windows venv python is a launcher: kill the whole tree
                p = psutil.Process(j["pid"])
                for c in p.children(recursive=True) + [p]:
                    c.terminate()
            except psutil.NoSuchProcess:
                pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["start", "status", "stop", "kill"])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--keep-alive", action="store_true", help="keep polling configs/queue/ for new jobs until stopped")
    a = ap.parse_args()
    if a.cmd == "start":
        conc = a.concurrency or yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml")))["queue"]["concurrency"]
        start(a.runs, conc, a.keep_alive)
    elif a.cmd == "status":
        print(open(STATUS).read() if os.path.exists(STATUS) else "no status")
    elif a.cmd == "stop":
        open(STOP, "w").close(); print("stop flag set")
    else:
        kill(); print("stop flag set; running jobs terminated")
    sys.exit(0)
