# DCNv4 reference (license only)

No DCNv4 code is vendored. The official operator (https://github.com/OpenGVLab/DCNv4, commit
4b848f7dd7da74ff03f7d278f902c6fd05b391b5, MIT License, copy in this folder) needs to compile CUDA kernels.
`pip install DCNv4` failed on this machine on 2026-11-04 with
`nvcc not found at 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin\nvcc.exe'`.

`src/operators.py::DCNv4Reimpl` is a PyTorch REIMPLEMENTATION written for this project, NOT the official
implementation. It follows the official module structure (value_proj, a single zero-initialised offset_mask
Linear with 3*K outputs padded to a multiple of 8, laid out per group as [2P offsets, P modulation], no softmax
on the modulation, output_proj) and the equivalence used by the official test script
`DCNv4_op/scripts/test_dcnv4.py` (DCNv4 output == DCNv3 core with un-normalised modulation). It does not
reproduce the official kernel's memory-access optimisations, so its speed says nothing about official DCNv4.
