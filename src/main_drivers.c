#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "matmat.h"
#include "matmat_auto.h"
#include "matvec.h"
#include "matvec_auto.h"
#include "sse_methods.h"
#include "main_drivers.h"
#include "util.h"

#define REPEATED_TIMES 200
#define GET_TIME(x); if(clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
{perror("clock_gettime(): "); exit(EXIT_FAILURE);}

// listing 7
// Matrix Matrix Drivers
void driveMatMatCPU(const float *mat_a, const float *mat_b, float *mat_c, int n, int cols) {
    struct timespec t0, t1;
    unsigned long sec, nsec;
    float mean;
    float times[REPEATED_TIMES];
    for (int i = 0; i < REPEATED_TIMES; ++i) {
        memset(mat_c, 0, sizeof(float) * n * cols);GET_TIME(t0);
        matmat_listing7(mat_c, n, n, mat_a, cols, mat_b);GET_TIME(t1);
        float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
        times[i] = el_t;
    }
    mean = Average(times, REPEATED_TIMES);
    printf("Average time : %f\n", mean);
}

// listing 7 sse
void driveMatMat_SSE(const float *mat_a, const float *mat_b, float *mat_c, int n, int cols) {
    struct timespec t0, t1;
    unsigned long sec, nsec;
    float mean;
    float times[REPEATED_TIMES];
    for (int i = 0; i < REPEATED_TIMES; ++i) {
        memset(mat_c, 0, sizeof(float) * n * cols);GET_TIME(t0);
        matmat_listing7_sse(mat_c, n, n, mat_a, cols, mat_b);GET_TIME(t1);
        float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
        times[i] = el_t;
    }
    mean = Average(times, REPEATED_TIMES);
    printf("Average time : %f\n", mean);
}


// listing 7 auto vectorizing 
void driveMatMatAuto(const float *mat_a, const float *mat_b, float *mat_c, int n, int cols) {
    struct timespec t0, t1;
    unsigned long sec, nsec;
    float mean;
    float times[REPEATED_TIMES];
    for (int i = 0; i < REPEATED_TIMES; ++i) {
        memset(mat_c, 0, sizeof(float) * n * cols);GET_TIME(t0);
        matmat_auto(mat_c, n, n, mat_a, cols, mat_b);GET_TIME(t1);
        float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
        times[i] = el_t;
    }
    mean = Average(times, REPEATED_TIMES);
    printf("Average time : %f\n", mean);
}

// Matrix Vector Drivers
void driveMatVecCPU_listing5(const float *mat, const float *vec_in, float *vec_out, int n) {
    struct timespec t0, t1;
    unsigned long sec, nsec;
    float mean;
    float times[REPEATED_TIMES];
    for (int i = 0; i < REPEATED_TIMES; ++i) {
        memset(vec_out, 0, sizeof(float) * n);GET_TIME(t0);
        matvec_simple_listing5(n, vec_out, mat, vec_in);GET_TIME(t1);
        float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
        times[i] = el_t;
    }
    mean = Average(times, REPEATED_TIMES);
    printf("Average time : %f\n", mean);
}

void driveMatVecCPU_listing6(const float *mat, const float *vec_in, float *vec_out, int n) {
    struct timespec t0, t1;
    unsigned long sec, nsec;
    float mean;
    float times[REPEATED_TIMES];
    for (int i = 0; i < REPEATED_TIMES; ++i) {
        memset(vec_out, 0, sizeof(float) * n);GET_TIME(t0);
        matvec_unrolled_listing6(n, vec_out, mat, vec_in);GET_TIME(t1);
        float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
        times[i] = el_t;
    }
    mean = Average(times, REPEATED_TIMES);
    printf("Average time : %f\n", mean);
}

// listing 6 sse
void driveMatVecSSE(const float *mat, const float *vec_in, float *vec_out, int n) {
    struct timespec t0, t1;
    unsigned long sec, nsec;
    float mean, sd;
    int initialCount = REPEATED_TIMES;
    float times[initialCount];
    for (int i = 0; i < initialCount; ++i) {
        memset(vec_out, 0, sizeof(float) * n);
        if (n >= 16) { GET_TIME(t0);
            matvec_unrolled_16sse(n, vec_out, mat, vec_in);GET_TIME(t1);
        } else { GET_TIME(t0);
            matvec_unrolled_sse_quite(n, vec_out, mat, vec_in);GET_TIME(t1);
        }
        float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
        times[i] = el_t;
    }
    mean = Average(times, initialCount);
    printf("Average time : %f\n", mean);
}

void driveMatVecAuto(const float *mat, const float *vec_in, float *vec_out, int n) {
    struct timespec t0, t1;
    unsigned long sec, nsec;
    float mean;
    float times[REPEATED_TIMES];
    for (int i = 0; i < REPEATED_TIMES; ++i) {
        memset(vec_out, 0, sizeof(float) * n);GET_TIME(t0);
        matvec_unrolled_auto(n, vec_out, mat, vec_in);GET_TIME(t1);
        float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
        times[i] = el_t;
    }
    mean = Average(times, REPEATED_TIMES);
    printf("Average time : %f\n", mean);
}