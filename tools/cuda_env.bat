@echo off
REM Per-shell CUDA + MSVC environment for this project only (no global changes).
REM Uses CUDA_HOME if already set (e.g. a system CUDA Toolkit >= 12.8); otherwise the project-local toolkit in
REM .cache\cuda\v12.8 (compiler components extracted from NVIDIA's official CUDA 12.8.1 installer).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if not defined CUDA_HOME set "CUDA_HOME=%~dp0..\.cache\cuda\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=12.0"
set DISTUTILS_USE_SDK=1
