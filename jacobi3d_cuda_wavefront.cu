#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "eval.h"

#define GET(X, Y, Z) gridOld[((X + dimX) % dimX) + ((Y + dimY) % dimY) * dimX + ((Z + dimZ) % dimZ) * dimX * dimY]
#define SET(X, Y, Z) gridNew[((X + dimX) % dimX) + ((Y + dimY) % dimY) * dimX + ((Z + dimZ) % dimZ) * dimX * dimY]

__global__ void update(double *gridOld, double *gridNew, int dimX, int dimY, int dimZ, int wavefrontLength)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int zStart = blockIdx.z * wavefrontLength;
    int zEnd = zStart + wavefrontLength;
    if (zEnd > dimZ) {
        zEnd = dimZ;
    }

    for (int z = zStart; z < zEnd; ++z) {
        SET(x, y, z) = (GET(x, y, z - 1) +
                        GET(x, y - 1, z) +
                        GET(x - 1, y, z) +
                        GET(x + 1, y, z) +
                        GET(x, y + 1, z) +
                        GET(x, y, z + 1)) * (1.0 / 6.0);
    }
}

void init(double *gridNew, int dimX, int dimY, int dimZ)
{
    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
                double value = 0;
                if ((x * y * z) == 0) {
                    value = 1;
                }
                SET(x, y, z) = value;
            }
        }
    }
}

void print(double *gridOld, int dimX, int dimY, int dimZ)
{
    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
                std::cout << " " << GET(x, y, z);
            }
            std::cout << "\n";
        }
    }
}

int divAndRoundUp(int dim, int blockDim)
{
    int res = dim / blockDim;
    if (dim % blockDim) {
        res += 1;
    }
    return res;
}

void checkForCUDAError()
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(error) << "\n";
        throw std::runtime_error("CUDA error");
    }
}

void benchmark(std::vector<double> *gridOld, std::vector<double> *gridNew, int dimX, int dimY, int dimZ, int repeats, dim3 blockDim, int wavefrontLength)
{
    checkForCUDAError();
    cudaDeviceSynchronize();
    double tStartInit = getUTtime();

    int byteSize = dimX * dimY * dimZ * sizeof(double);
    double *devGridOld;
    double *devGridNew;
    cudaMalloc(&devGridOld, byteSize);
    cudaMalloc(&devGridNew, byteSize);
    cudaMemcpy(devGridOld, &gridOld->front(), byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devGridNew, &gridNew->front(), byteSize, cudaMemcpyHostToDevice);
    dim3 gridDim(divAndRoundUp(dimX, blockDim.x),
                 divAndRoundUp(dimY, blockDim.y),
                 divAndRoundUp(dimZ, blockDim.z * wavefrontLength));

    cudaDeviceSynchronize();
    double tStartCalc = getUTtime();

    for (int t = 0; t < repeats; ++t) {
        update<<<gridDim, blockDim>>>(devGridOld, devGridNew, dimX, dimY, dimZ, wavefrontLength);
        std::swap(devGridOld, devGridNew);
    }

    cudaDeviceSynchronize();
    double tEndCalc = getUTtime();

    cudaMemcpy(&gridOld->front(), devGridOld, byteSize, cudaMemcpyDeviceToHost);
    cudaFree(devGridOld);
    cudaFree(devGridNew);

    cudaDeviceSynchronize();
    double tEnd = getUTtime();
    checkForCUDAError();
    eval(tStartInit, tStartCalc, tEndCalc, tEnd, dimX, dimY, dimZ, repeats);
}

int main(int argc, char **argv)
{
    if ((argc < 6) || (argc > 9)) {
        std::cerr << "usage: " << argv[0] << " DIM_X DIM_Y DIM_Z REPEATS CUDA_DEVICE [BLOCK_DIM_X=32] [BLOCK_DIM_Y=32] [WAVEFRONT_LENGTH=1] \n";
        return 1;
    }
    std::stringstream buf;
    for (int i = 1; i < argc; ++i) {
        buf << argv[i] << " ";
    }
    int dimX, dimY, dimZ, repeats, cudaDevice, wavefrontLength;
    buf >> dimX;
    buf >> dimY;
    buf >> dimZ;
    buf >> repeats;
    buf >> cudaDevice;
    cudaSetDevice(cudaDevice);
    dim3 blockDim(32, 32, 1);

    if (argc > 5) {
        buf >> blockDim.x;
    }

    if (argc > 6) {
        buf >> blockDim.y;
    }

    if (argc > 7) {
        buf >> wavefrontLength;
    }

    int size = dimX * dimY * dimZ;
    std::vector<double> gridOld(size);
    std::vector<double> gridNew(size);
    init(&gridOld[0], dimX, dimY, dimZ);
    init(&gridNew[0], dimX, dimY, dimZ);

    benchmark(&gridOld, &gridNew, dimX, dimY, dimZ, repeats, blockDim, wavefrontLength);

    print(&gridOld[0], dimX, dimY, dimZ);
}
