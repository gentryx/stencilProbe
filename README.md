stencilProbe
============

Comparison of performance/complexity of stencil code implementations
for different architectures. Metrics: Performance (measured in GLUPS)
and code complexity.

1. Rationale
============

- Naming Scheme: BENCHMARK_ARCHITECTURE_VARIANT.SUFFIX
  e.g. jacobi3d_opensse_vanilla.cpp.

- Architectures:
  - serial = standard, sequential code
  - openmp = parallelized with OpenMP
  - cuda = manually parallelized with CUDA

- Variants:
  - vanilla: straightforward code
  - avx: manually vectorized code with AVX intrinsics

- Benchmarks:
  - jacobi3d: simple 3D Jacobi iteration

2. Compiling/Running
====================

stencilProbe comes with a standard CMake script, configuration of the compiler is hence done with "-D CMAKE_CXX_COMPILER=..." and "-D CMAKE_CXX_FLAGS=...".

  cd stencilProbe
  mkdir build
  cd build
  cmake ../
  make -j10
  ./jacobi3d_serial_vanilla 128 128 128 10 >/dev/null

3. (not so) FAQ
===============

Q: All those codes are horribly redundant. Why?

A: The purpose of the codes is to track which (minimal) modifications
   to a sequential, vanilla code are required to implement certain
   features.
