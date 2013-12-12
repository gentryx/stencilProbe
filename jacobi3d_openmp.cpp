#include <iostream>
#include <sstream>
#include <vector>

#include "eval.h"

#define GET(X, Y, Z) gridOld[(X) + (Y) * dimX + (Z) * dimX * dimY]
#define SET(X, Y, Z) gridNew[(X) + (Y) * dimX + (Z) * dimX * dimY]

void update(double *gridOld, double *gridNew, int dimX, int dimY, int dimZ)
{
#pragma omp parallel for
    for (int z = 1; z < (dimZ - 1); ++z) {
        for (int y = 1; y < (dimY - 1); ++y) {
            for (int x = 1; x < (dimX - 1); ++x) {
                SET(x, y, z) = (GET(x, y, z - 1) +
                                GET(x, y - 1, z) +
                                GET(x - 1, y, z) +
                                GET(x + 1, y, z) +
                                GET(x, y + 1, z) +
                                GET(x, y, z + 1)) * (1.0 / 6.0);
            }
        }
    }
}

void init(double *gridNew, int dimX, int dimY, int dimZ)
{
#pragma omp parallel for
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

void benchmark(std::vector<double> *gridOld, std::vector<double> *gridNew, int dimX, int dimY, int dimZ, int repeats)
{
    double tStartInit = getUTtime();
    double tStartCalc = getUTtime();

    for (int t = 0; t < repeats; ++t) {
        update(&gridOld->front(), &gridNew->front(), dimX, dimY, dimZ);
        std::swap(gridOld, gridNew);
    }

    double tEndCalc = getUTtime();
    double tEnd = getUTtime();
    eval(tStartInit, tStartCalc, tEndCalc, tEnd, dimX, dimY, dimZ, repeats);
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        std::cerr << "usage: " << argv[0] << " DIM_X DIM_Y DIM_Z REPEATS\n";
        return 1;
    }
    std::stringstream buf;
    for (int i = 1; i < argc; ++i) {
        buf << argv[i] << " ";
    }
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
