#ifndef EVAL_H
#define EVAL_H

#include <sys/time.h>

double getUTtime()
{
    timeval t;
    gettimeofday(&t, 0);
    return t.tv_sec + t.tv_usec * 1e-6;
}

void eval(double tStartInit, double tStartCalc, double tEndCalc, double tEnd, int dimX, int dimY, int dimZ, int repeats)
{
    double tNetto = tEndCalc - tStartCalc;
    double tBrutto = tEnd - tStartInit;
    double tOverhead = tBrutto - tNetto;

    double gUpdates = 1e-9 * dimX * dimY * dimZ * repeats;
    double glupsNetto = gUpdates / tNetto;
    double glupsBrutto = gUpdates / tBrutto;

    std::cerr << "gridDim: (" << dimX << ", " << dimY << ", " << dimZ << ")\n"
              << "repeats: " << repeats << "\n"
              << "compute time: " << tNetto << " s\n"
              << "overhead time: " << tOverhead << " s\n"
              << "GLUPS peak: " << glupsNetto << "\n"
              << "GLUPS: " << glupsBrutto << "\n";
}

#endif
