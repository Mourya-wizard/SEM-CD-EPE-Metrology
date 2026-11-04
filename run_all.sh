#!/usr/bin/env bash
# Reproduce the project from a clean clone.  Usage:  bash run_all.sh [--smoke]
#   --smoke : environment + tests + a few-minute end-to-end pass on 2+2 validation tiles (needs data/ present
#             or PROJECT_DATA_ROOT pointing at an existing data/ folder; downloads nothing large).
#   (full)  : downloads data (~15 GB), runs M1..M6 + M8 benchmark prerequisites. Training runs through the GPU
#             queue and takes several GPU-hours. Steps are resumable (cached files are skipped).
# Works under Git Bash on Windows (.venv/Scripts) and on Linux (.venv/bin). Never installs outside .venv.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
SMOKE=0; [[ "${1:-}" == "--smoke" ]] && SMOKE=1

export HF_HOME="$ROOT/.cache/huggingface" TORCH_HOME="$ROOT/.cache/torch" XDG_CACHE_HOME="$ROOT/.cache" MPLCONFIGDIR="$ROOT/.cache/mpl"
export PIP_REQUIRE_VIRTUALENV=1 PIP_CACHE_DIR="$ROOT/.cache/pip" CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# ---------------------------------------------------------------------------------------------- venv
if [[ -x .venv/Scripts/python.exe ]]; then PY=.venv/Scripts/python.exe
elif [[ -x .venv/bin/python ]]; then PY=.venv/bin/python
else
  SYS_PY="$(command -v python3 || command -v python)"
  "$SYS_PY" -m venv .venv
  if [[ -x .venv/Scripts/python.exe ]]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements.txt
fi
echo "[run_all] python: $PY"
"$PY" -c "import torch; print('[run_all] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ---------------------------------------------------------------------------------------------- data
LB="${PROJECT_DATA_ROOT:-$ROOT/data}/lithobench"
CA="${PROJECT_DATA_ROOT:-$ROOT/data}/carinthia"
if [[ ! -d "$LB/MetalSet" ]]; then
  if [[ $SMOKE == 1 ]]; then echo "[run_all] smoke needs LithoBench in $LB (set PROJECT_DATA_ROOT)"; exit 1; fi
  mkdir -p "$LB"
  "$PY" -m gdown "https://drive.google.com/uc?id=1MzYiRRxi8Eu2L6WHCfZ1DtRnjVNOl4vu" -O "$LB/lithodata.tar.gz"
  (cd "$LB" && tar xzf lithodata.tar.gz)
fi
if [[ ! -f "$CA/data/carinthia-s.csv" && $SMOKE == 0 ]]; then
  mkdir -p "$CA"
  curl -sSL -o "$CA/data.zip" "https://zenodo.org/records/16895427/files/data.zip?download=1"
  (cd "$CA" && unzip -q -o data.zip)
  curl -sSL -o "$CA/carinthia-s_dataset.html" "https://zenodo.org/records/16895427/files/carinthia-s_dataset.html?download=1"
fi

# ---------------------------------------------------------------------------------------------- smoke
if [[ $SMOKE == 1 ]]; then
  [[ -f results/splits.json ]] || "$PY" src/data_lithobench.py --audit
  "$PY" -m pytest -q tests/test_sim_truth.py tests/test_registration.py tests/test_operators.py
  if compgen -G "build/edge_refine_ext*" >/dev/null; then "$PY" -m pytest -q tests/test_kernel.py; else echo "[run_all] kernel not built (cuda/build.bat), skipping tests/test_kernel.py"; fi
  "$PY" src/evaluate.py cache --split smoke
  "$PY" src/evaluate.py register --split smoke
  "$PY" tools/smoke_check.py
  if [[ -f data/bank/meta.json || -n "${PROJECT_DATA_ROOT:-}" && -f "${PROJECT_DATA_ROOT:-}/bank/meta.json" ]]; then
    mkdir -p configs/runs/smoke
    printf 'name: smoke_run_all\nop: conv\ncad: true\nseed: 1\ntrain: {iters: 30, val_every: 30, val_tiles_per_set: 2, val_split: smoke, log_every: 10, warmup: 5}\n' > configs/runs/smoke/smoke_run_all.yaml
    rm -rf checkpoints/smoke_run_all
    "$PY" tools/run_queue.py start --runs configs/runs/smoke/smoke_run_all.yaml --concurrency 1
    "$PY" -c "import json; s=json.load(open('checkpoints/smoke_run_all/status.json')); print('[run_all] smoke train', s['state'], s['best']); assert s['state']=='done'"
  else
    echo "[run_all] no crop bank (src/train.py --build-bank), skipping smoke training"
  fi
  echo "[run_all] SMOKE OK"
  exit 0
fi

# ---------------------------------------------------------------------------------------------- full
"$PY" src/data_lithobench.py --audit
"$PY" src/carinthia.py --audit
"$PY" -m pytest -q tests/test_sim_truth.py tests/test_registration.py tests/test_operators.py
"$PY" src/sem_sim.py --m2-report
for split in val test; do "$PY" src/evaluate.py cache --split $split; "$PY" src/evaluate.py register --split $split; done
"$PY" src/evaluate.py baseline
"$PY" src/plots.py baseline
[[ -f data/bank/meta.json ]] || "$PY" src/train.py --build-bank
"$PY" tools/gpu_pilot.py --op dcnv3
"$PY" tools/op_table.py
# Training + evaluation through the GPU queue (resumable; run matrix in configs/runs/).
RUNS=$(ls configs/runs/screen_*.yaml configs/runs/confirm_*.yaml configs/runs/cadtest_*.yaml)
"$PY" tools/run_queue.py start --runs $RUNS
echo "[run_all] training done. Evaluations: drop eval yamls into configs/queue/ (see docs/ARCHITECTURE.md), then"
echo "          src/evaluate.py / src/confidence.py / src/plots.py as listed in README.md 'Exact commands'."
