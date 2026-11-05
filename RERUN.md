# Rerunning the project

Everything in this repository can be regenerated. Datasets, simulated SEM sets, training data and model weights are
not stored in git; the commands below download or rebuild them. All commands run from the repository root in `bash`
(Git Bash on Windows). Use `.venv/Scripts/python.exe` on Windows and `.venv/bin/python` on Linux; it is written
`$PY` below.

## Prerequisites
- Python 3.10, `bash`, `curl`
- NVIDIA GPU with a CUDA 12.8-compatible driver (tested on an RTX PRO 5000 Blackwell, sm_120)
- About 60 GB free disk (LithoBench 39 GB while extracting, 24 GB after deleting the archive; simulated sets 7 GB; training crop bank 13 GB)
- Optional, for the C++/CUDA kernel: `nvcc` ≥ 12.8 and a C++ compiler (MSVC on Windows). Set `CUDA_HOME` if the
  toolkit is not on `PATH`.

```bash
python -m venv .venv                       # Windows: py -3.10 -m venv .venv
$PY -m pip install -r requirements.txt
```

## Option 1: one command
```bash
bash run_all.sh            # full rerun, ~12 hours on one GPU, resumable
bash run_all.sh --smoke    # ~1 minute check once the datasets are present
```
The full run downloads both datasets, rebuilds the simulated validation/test sets, runs the classical baseline,
trains every model through the GPU job queue, evaluates them, runs the real-SEM pipeline and the trust analysis,
builds and checks the CUDA kernel, and regenerates every table and figure. Each step skips outputs that already
exist, so an interrupted run continues where it stopped.

## Option 2: step by step
| Step | Command | Output | Time |
|---|---|---|---|
| 1. Datasets | first part of `bash run_all.sh` (stop after the downloads), or see *Manual download* | `data/lithobench/`, `data/carinthia/` | ~15 min |
| 2. Data audit and splits | `$PY src/data_lithobench.py --audit` | `results/splits.json` (already in git) | 1 min |
| 3. Simulator checks | `$PY src/sem_sim.py --m2-report` | `results/simulator_checks/` | 1 min |
| 4. Simulated val/test sets | `$PY src/evaluate.py cache --split val` and `--split test` | `results/raw/cache/` | ~6 min |
| 5. Registration | `$PY src/evaluate.py register --split val` and `--split test` | `results/raw/reg/` | ~20 min |
| 6. Classical baseline | `$PY src/evaluate.py baseline` | `results/classical_baseline/` | ~10 min |
| 7. Training crop bank | `$PY src/train.py --build-bank` | `data/bank/` | ~3 min |
| 8. Training | `$PY tools/run_queue.py start --runs configs/runs/screen_*.yaml configs/runs/confirm_*.yaml configs/runs/cadtest_*.yaml` | `checkpoints/<run>/` | ~6 GPU-hours |
| 9. Evaluation, real SEM, trust analysis, figures | `$PY tools/reproduce_results.py` | `results/`, `docs/RESULTS.md` | ~5 hours |
| 10. CUDA kernel | `cuda\build.bat` (Windows), then `$PY -m pytest tests/test_kernel.py` and `$PY benchmarks/bench_kernel.py parity` | `build/`, `results/cuda_kernel/` | ~2 min |

The timing benchmark (`benchmarks/bench_kernel.py bench`) needs exclusive use of the GPU: create
`results/gpu_exclusive.lock`, make sure no queue job is running, run it, then delete the lock.

## Option 3: re-evaluate trained models without retraining
If `checkpoints/` already holds the trained models, skip steps 7–8: run steps 1, 4 and 5, then step 9.
`tools/reproduce_results.py` evaluates every checkpoint it finds and only computes what is missing.

## Manual download (if an automatic download fails)
- **LithoBench** (14.7 GB): download `lithodata.tar.gz` from the Google Drive link in the
  [LithoBench README](https://github.com/shelljane/lithobench) into `data/lithobench/`, then
  `cd data/lithobench && tar xzf lithodata.tar.gz`. Google Drive sometimes limits downloads of large files; retry later or
  download through a browser.
- **Carinthia-S** (139 MB): `curl -L -o data/carinthia/data.zip "https://zenodo.org/records/16895427/files/data.zip?download=1"`
  then `cd data/carinthia && unzip data.zip`.

## Checks
```bash
$PY -m pytest tests          # 45 tests; 21 need the datasets, 7 need the CUDA extension
bash run_all.sh --smoke      # end-to-end pass on 4 tiles
$PY tools/run_queue.py status
```

## Notes
- On Windows, `python3` may resolve to the Microsoft Store stub; create the environment with `py -3.10 -m venv .venv`.
- CPU-heavy steps use at most `eval.max_cpu_workers` processes (`configs/default.yaml`); training runs one job at a
  time because a concurrency pilot showed no throughput gain from running two.
- Results are deterministic for a given seed; reruns reproduce the numbers in `results/` up to GPU floating-point
  nondeterminism in training.
