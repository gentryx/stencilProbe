#include <immintrin.h>
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
            int x = 1;
            for (; x < 4; ++x) {
                SET(x, y, z) = (GET(x, y, z - 1) +
                                GET(x, y - 1, z) +
                                GET(x - 1, y, z) +
                                GET(x + 1, y, z) +
                                GET(x, y + 1, z) +
                                GET(x, y, z + 1)) * (1.0 / 6.0);
            }

            __m256d oneSixth = _mm256_set1_pd(1.0 / 6.0);

            for (; x < (dimX - 16); x += 16) {
                // load south
                __m256d sum0 = _mm256_load_pd(&GET(x +  0, y, z - 1));
                __m256d sum1 = _mm256_load_pd(&GET(x +  4, y, z - 1));
                __m256d sum2 = _mm256_load_pd(&GET(x +  8, y, z - 1));
                __m256d sum3 = _mm256_load_pd(&GET(x + 12, y, z - 1));

                // load top
                __m256d buf0 = _mm256_load_pd(&GET(x +  0, y - 1, z));
                __m256d buf1 = _mm256_load_pd(&GET(x +  4, y - 1, z));
                __m256d buf2 = _mm256_load_pd(&GET(x +  8, y - 1, z));
                __m256d buf3 = _mm256_load_pd(&GET(x + 12, y - 1, z));

                // load west
                __m256d bufA = _mm256_loadu_pd(&GET(x -  1, y, z));
                __m256d bufB = _mm256_loadu_pd(&GET(x +  3, y, z));
                __m256d bufC = _mm256_loadu_pd(&GET(x +  7, y, z));
                __m256d bufD = _mm256_loadu_pd(&GET(x + 11, y, z));

                // add south and top
                sum0 = _mm256_add_pd(sum0, buf0);
                sum1 = _mm256_add_pd(sum1, buf1);
                sum2 = _mm256_add_pd(sum2, buf2);
                sum3 = _mm256_add_pd(sum3, buf3);

                // load east
                buf0 = _mm256_loadu_pd(&GET(x +  1, y, z));
                buf1 = _mm256_loadu_pd(&GET(x +  5, y, z));
                buf2 = _mm256_loadu_pd(&GET(x +  9, y, z));
                buf3 = _mm256_loadu_pd(&GET(x + 13, y, z));

                // add (south+top) and west
                sum0 = _mm256_add_pd(sum0, bufA);
                sum1 = _mm256_add_pd(sum1, bufB);
                sum2 = _mm256_add_pd(sum2, bufC);
                sum3 = _mm256_add_pd(sum3, bufD);

                // load bottom
                bufA = _mm256_load_pd(&GET(x +  0, y + 1, z));
                bufB = _mm256_load_pd(&GET(x +  4, y + 1, z));
                bufC = _mm256_load_pd(&GET(x +  8, y + 1, z));
                bufD = _mm256_load_pd(&GET(x + 12, y + 1, z));

                // add (south+top+west) and east
                sum0 = _mm256_add_pd(sum0, buf0);
                sum1 = _mm256_add_pd(sum1, buf1);
                sum2 = _mm256_add_pd(sum2, buf2);
                sum3 = _mm256_add_pd(sum3, buf3);

                // load north
                buf0 = _mm256_load_pd(&GET(x +  0, y, z + 1));
                buf1 = _mm256_load_pd(&GET(x +  4, y, z + 1));
                buf2 = _mm256_load_pd(&GET(x +  8, y, z + 1));
                buf3 = _mm256_load_pd(&GET(x + 12, y, z + 1));

                // add (south+top+west+east) and bottom
                sum0 = _mm256_add_pd(sum0, bufA);
                sum1 = _mm256_add_pd(sum1, bufB);
                sum2 = _mm256_add_pd(sum2, bufC);
                sum3 = _mm256_add_pd(sum3, bufD);

                // add (south+top+west+east+bottom) and north
                sum0 = _mm256_add_pd(sum0, buf0);
                sum1 = _mm256_add_pd(sum1, buf1);
                sum2 = _mm256_add_pd(sum2, buf2);
                sum3 = _mm256_add_pd(sum3, buf3);

                // scale down
                sum0 = _mm256_mul_pd(sum0, oneSixth);
                sum1 = _mm256_mul_pd(sum1, oneSixth);
                sum2 = _mm256_mul_pd(sum2, oneSixth);
                sum3 = _mm256_mul_pd(sum3, oneSixth);
            }

            for (; x < (dimX - 1); ++x) {
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
    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
                double value = 1;
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
    for (int i = 1; i <= 4; ++i)
        buf << argv[i] << " ";
    int dimX, dimY, dimZ, repeats;
    buf >> dimX;
    buf >> dimY;
    buf >> dimZ;
    buf >> repeats;

    if ((dimX < 4) || ((dimX % 4) != 0)) {
        std::cerr << "DIM_X needs to be > 4 and divisible by 4";
    }

    int size = dimX * dimY * dimZ;
    std::vector<double> gridOld(size);
    std::vector<double> gridNew(size);
    init(&gridOld[0], dimX, dimY, dimZ);
    init(&gridNew[0], dimX, dimY, dimZ);

    benchmark(&gridOld, &gridNew, dimX, dimY, dimZ, repeats);

    print(&gridOld[0], dimX, dimY, dimZ);
}
