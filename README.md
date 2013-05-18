stencilProbe
============

Comparison of performance/complexity of stencil code implementations for different architectures

- naming scheme: BENCHMARK_ARCHITECTURE.cpp/cu, e.g. jacobi3d_opensse.cpp
- architectures:
  - vanilla = standard C++,
  - openmpsse = parallelized with OpenMP and manually vectorized
  - cuda = manually parallelized with CUDA
- benchmarks:
  - jacobi3d: simple 3D Jacobi iteration
- run:
