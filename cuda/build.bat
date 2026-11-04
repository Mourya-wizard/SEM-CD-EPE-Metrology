@echo off
REM Build the pybind11 + CUDA edge-refinement extension into build\ (project-local toolchain, no system changes).
REM Usage from the repo root:  cmd //c cuda\build.bat   (Git Bash)   or   cuda\build.bat   (cmd)
setlocal
set "ROOT=%~dp0.."
call "%ROOT%\tools\cuda_env.bat" || exit /b 1
cd /d "%ROOT%"
"%ROOT%\.venv\Scripts\cmake.exe" -S cuda -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MAKE_PROGRAM="%ROOT%\.venv\Scripts\ninja.exe" -DPython_EXECUTABLE="%ROOT%\.venv\Scripts\python.exe" ^
  -Dpybind11_DIR="%ROOT%\.venv\Lib\site-packages\pybind11\share\cmake\pybind11" ^
  -DCMAKE_CUDA_COMPILER="%CUDA_HOME%\bin\nvcc.exe" || exit /b 1
"%ROOT%\.venv\Scripts\cmake.exe" --build build -j 4 || exit /b 1
