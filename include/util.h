//
// Created by jawadhsr on 2/27/17.
//
#include <time.h>
#include <xmmintrin.h>
#ifndef ACA_LAB3_UTIL_H
#define ACA_LAB3_UTIL_H

#define XMM_ALIGNMENT_BYTES 16

void matrixCreationNByN_1D(int r, int c, float **mat_a);

void printNByCMat(const float *mat, int n, int c);

void printVector(const float *vec, int n);
float elapsed_time_microsec(struct timespec *begin, struct timespec *end,
                            unsigned long *sec, unsigned long *nsec);
float standardDeviation(float times[], int numSamples);
float Average(float *times, int numSamples);
long requiredSampleSize(float sd, float mean);

void print_vector_ps(__m128 v);

int getArguments(int argc, char *argv[], int *n, short *mat_vec_ver, short *mat_mat_ver, short *c_ver, short *sse_ver,
                 short *a_vec_ver, short *test, short *listing6);
#endif //ACA_LAB3_UTIL_H
