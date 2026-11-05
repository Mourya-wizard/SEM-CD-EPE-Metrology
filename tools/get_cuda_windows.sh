#!/usr/bin/env bash
# Project-local CUDA 12.8 compiler for the C++/CUDA kernel on Windows: no admin rights, no driver change.
# Downloads NVIDIA's official installer (3.4 GB, MD5-checked), extracts only the compiler components into
# .cache/cuda/v12.8 (~2.5 GB, used by tools/cuda_env.bat), then deletes the installer. No-op if already present.
set -euo pipefail
cd "$(dirname "$0")/.."
DST=.cache/cuda/v12.8
if [[ -x "$DST/bin/nvcc.exe" ]]; then echo "[cuda] compiler already present in $DST"; exit 0; fi
TMP=.cache/cuda_dl
mkdir -p "$TMP"
curl -fL --retry 3 -o "$TMP/cuda.exe" \
  https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_572.61_windows.exe
echo "05c59d04ac6c63686c6a0498e472c1e7  $TMP/cuda.exe" | md5sum -c -   # NVIDIA's published md5sum.txt
(cd "$TMP" && /c/Windows/System32/tar.exe -xf cuda.exe cuda_nvcc/nvcc cuda_cudart/cudart cuda_cccl/thrust \
  cuda_profiler_api/cuda_profiler_api cuda_nvrtc cuda_nvtx)
mkdir -p "$DST"
for d in "$TMP"/*/*/; do for s in bin include lib nvvm; do [[ -d "$d$s" ]] && cp -r "$d$s" "$DST/"; done; done
rm -rf "$TMP"
"$DST/bin/nvcc.exe" --version | tail -1
