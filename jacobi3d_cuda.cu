#include <iostream>
#include <sstream>
#include <vector>

#define GET(X, Y, Z) gridOld[(X) + (Y) * dimX + (Z) * dimX * dimY]
#define SET(X, Y, Z) gridNew[(X) + (Y) * dimX + (Z) * dimX * dimY]

__global__ void update(double *gridOld, double *gridNew, int dimX, int dimY, int dimZ)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if ((x == 0) || (x >= (dimX - 1)) ||
        (y == 0) || (y >= (dimY - 1)) ||
        (z == 0) || (z >= (dimZ - 1))) {
        return;
    }

    SET(x, y, z) = (GET(x, y, z - 1) +
                    GET(x, y - 1, z) +
                    GET(x - 1, y, z) +
                    GET(x + 1, y, z) +
                    GET(x, y + 1, z) +
                    GET(x, y, z + 1)) * (1.0 / 6.0);
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

void benchmark(std::vector<double> *gridOld, std::vector<double> *gridNew, int dimX, int dimY, int dimZ, int repeats)
{
    int byteSize = dimX * dimY * dimZ * sizeof(double);
    double *devGridOld;
    double *devGridNew;
    cudaMalloc(&devGridOld, byteSize);
    cudaMalloc(&devGridNew, byteSize);
    cudaMemcpy(devGridOld, &gridOld->front(), byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devGridNew, &gridNew->front(), byteSize, cudaMemcpyHostToDevice);
    dim3 blockDim(32, 32, 1);
    dim3 gridDim(divAndRoundUp(dimX, blockDim.x),
                 divAndRoundUp(dimY, blockDim.y),
                 divAndRoundUp(dimZ, blockDim.z));

    for (int t = 0; t < repeats; ++t) {
        update<<<gridDim, blockDim>>>(devGridOld, devGridNew, dimX, dimY, dimZ);
        std::swap(devGridOld, devGridNew);
    }

    cudaMemcpy(&gridOld->front(), devGridOld, byteSize, cudaMemcpyDeviceToHost);
    cudaFree(devGridOld);
    cudaFree(devGridNew);
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        std::cerr << "usage: " << argv[0] << " DIM_X DIM_Y DIM_Z REPEATS\n";
        return 1;
    }
    std::stringstream buf;
    for (int i = 1; i <= 4; ++i)
        buf << argv[i] << " ";
    int dimX, dimY, dimZ, repeats;
    buf >> dimX;
    buf >> dimY;
    buf >> dimZ;
    buf >> repeats;

    int size = dimX * dimY * dimZ;
    std::vector<double> gridOld(size);
    std::vector<double> gridNew(size);
    init(&gridOld[0], dimX, dimY, dimZ);
    init(&gridNew[0], dimX, dimY, dimZ);

    benchmark(&gridOld, &gridNew, dimX, dimY, dimZ, repeats);

    print(&gridOld[0], dimX, dimY, dimZ);
}