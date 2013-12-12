cmake_minimum_required(VERSION 2.8)

find_package(CUDA)

set(CMAKE_CXX_FLAGS "-fopenmp -march=native -O3")

if(CUDA_FOUND)
  cuda_add_executable(jacobi3d_cuda_vanilla jacobi3d_cuda_vanilla.cu)
endif()

add_executable(jacobi3d_vanilla   jacobi3d_vanilla.cpp)
add_executable(jacobi3d_openmp    jacobi3d_openmp.cpp)
add_executable(jacobi3d_openmpavx jacobi3d_openmpavx.cpp)
