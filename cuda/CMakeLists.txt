cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)          # RTX PRO 5000 Blackwell (sm_120); must be set before project()
endif()
project(edge_refine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)       # no cudart DLL needed next to the .pyd

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

pybind11_add_module(edge_refine_ext python_binding.cpp edge_refine_cpu.cpp edge_refine.cu)
target_link_libraries(edge_refine_ext PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(edge_refine_ext PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:$<IF:$<CXX_COMPILER_ID:MSVC>,/O2 /fp:precise,-O3>>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)   # no --use_fast_math: keep IEEE single precision
# put the .pyd directly in the build dir so Python can import it with sys.path.insert(0, "build")
set_target_properties(edge_refine_ext PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}")
