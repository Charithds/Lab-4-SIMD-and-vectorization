//
// Created by jawadhsr on 2/27/17.
//
#include <time.h>
#include <xmmintrin.h>
#ifndef ACA_LAB3_UTIL_H
#define ACA_LAB3_UTIL_H

#define XMM_ALIGNMENT_BYTES 32

void matrixCreationNByN_1D(int r, int c, float **mat_a);
double elapsed_time(clock_t tic, clock_t toc);
double Average(double *times, int numSamples);

int getArguments(int argc, char *argv[], int *n);
#endif
