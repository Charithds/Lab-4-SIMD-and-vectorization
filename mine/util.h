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

int getArguments(int argc, char *argv[], int *n, short *mat_vec_ver, short *mat_mat_ver, short *c_ver, short *sse_ver,
                 short *a_vec_ver, short *test, short *listing6);
#endif
