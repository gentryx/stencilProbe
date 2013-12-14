cmake_minimum_required(VERSION 2.8)

find_package(CUDA)

set(CMAKE_CXX_FLAGS "-fopenmp -march=native -O3")

if(CUDA_FOUND)
  cuda_add_executable(jacobi3d_cuda_vanilla           jacobi3d_cuda_vanilla.cu)
  cuda_add_executable(jacobi3d_cuda_wavefront         jacobi3d_cuda_wavefront.cu)
  cuda_add_executable(jacobi3d_cuda_wavefrontunrolled jacobi3d_cuda_wavefrontunrolled.cu)
endif()

add_executable(jacobi3d_serial_vanilla jacobi3d_serial_vanilla.cpp)
add_executable(jacobi3d_openmp_vanilla jacobi3d_openmp_vanilla.cpp)
add_executable(jacobi3d_openmp_avx     jacobi3d_openmp_avx.cpp)
