//
// Created by krv on 3/10/17.
//
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


#define GET_TIME(x); if(clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
{perror("clock_gettime(): "); exit(EXIT_FAILURE);}

// Matrix Matrix Drivers
void driveMatMatCPU(const float *mat_a, const float *mat_b, float *mat_c, int n, int cols) {
    struct timespec t0, t1;
    unsigned long sec, nsec;
    float mean, sd;
    float times[10];
    int req_n;
    for (int i = 0; i < 10; ++i) {
        memset(mat_c, 0, sizeof(float) * n * cols);GET_TIME(t0);
        matmat_listing7(mat_c, n, n, mat_a, cols, mat_b);GET_TIME(t1);
        float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
        times[i] = el_t;
    }
    mean = Average(times, 10);
    sd = standardDeviation(times, 10);
    req_n = requiredSampleSize(sd, mean);
    printf("Average time : %f\n", mean);
    printf("Required sample size for 95 percent confidence : %d\n", req_n);
    if (req_n > 10) {
        float *times_2 = (float *) malloc(sizeof(float) * req_n);
        printf("Automatically running with required sample size");
        for (int i = 0; i < req_n; ++i) {
            memset(mat_c, 0, sizeof(float) * n * cols);GET_TIME(t0);
            matmat_listing7(mat_c, n, n, mat_a, cols, mat_b);GET_TIME(t1);
            float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
            times_2[i] = el_t;
        }
        mean = Average(times_2, req_n);
        sd = standardDeviation(times_2, req_n);
        req_n = requiredSampleSize(sd, mean);
        printf("Average time : %f\n", mean);
        free(times_2);
    }
}

void driveMatMat_SSE(const float *mat_a, const float *mat_b, float *mat_c, int n, int cols) {
    struct timespec t0, t1;
    unsigned long sec, nsec;
    float mean, sd;
    float times[10];
    int req_n;
    for (int i = 0; i < 10; ++i) {
        memset(mat_c, 0, sizeof(float) * n * cols);GET_TIME(t0);
        matmat_listing7_sse(mat_c, n, n, mat_a, cols, mat_b);GET_TIME(t1);
        float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
        times[i] = el_t;
    }
    mean = Average(times, 10);
    sd = standardDeviation(times, 10);
    req_n = requiredSampleSize(sd, mean);
    printf("Average time : %f\n", mean);
    printf("Required sample size for 95 percent confidence : %d\n", req_n);
    if (req_n > 10) {
        float *times_2 = (float *) malloc(sizeof(float) * req_n);
        printf("Automatically running with required sample size");
        for (int i = 0; i < req_n; ++i) {
            memset(mat_c, 0, sizeof(float) * n * cols);GET_TIME(t0);
            matmat_listing7_sse(mat_c, n, n, mat_a, cols, mat_b);GET_TIME(t1);
            float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
            times_2[i] = el_t;
        }
        mean = Average(times_2, req_n);
        sd = standardDeviation(times_2, req_n);
        req_n = requiredSampleSize(sd, mean);
        printf("Average time : %f\n", mean);
        free(times_2);
    }
}

void driveMatMatAuto(const float *mat_a, const float *mat_b, float *mat_c, int n, int cols) {
    struct timespec t0, t1;
    unsigned long sec, nsec;
    float mean, sd;
    float times[10];
    int req_n;
    for (int i = 0; i < 10; ++i) {
        memset(mat_c, 0, sizeof(float) * n * cols);GET_TIME(t0);
        matmat_auto(mat_c, n, n, mat_a, cols, mat_b);GET_TIME(t1);
        float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
        times[i] = el_t;
    }
    mean = Average(times, 10);
    sd = standardDeviation(times, 10);
    req_n = requiredSampleSize(sd, mean);
    printf("Average time : %f\n", mean);
    printf("Required sample size for 95 percent confidence : %d\n", req_n);
    if (req_n > 10) {
        float *times_2 = (float *) malloc(sizeof(float) * req_n);
        printf("Automatically running with required sample size");
        for (int i = 0; i < req_n; ++i) {
            memset(mat_c, 0, sizeof(float) * n * cols);GET_TIME(t0);
            matmat_auto(mat_c, n, n, mat_a, cols, mat_b);GET_TIME(t1);
            float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
            times_2[i] = el_t;

        }
        mean = Average(times_2, req_n);
        sd = standardDeviation(times_2, req_n);
        req_n = requiredSampleSize(sd, mean);
        printf("Average time : %f\n", mean);
        free(times_2);
    }
}

// Matrix Vector Drivers
void driveMatVecCPU_listing5(const float *mat, const float *vec_in, float *vec_out, int n) {
    struct timespec t0, t1;
    unsigned long sec, nsec;
    float mean, sd;
    float times[10];
    int req_n = 10;
    for (int i = 0; i < 10; ++i) {
        memset(vec_out, 0, sizeof(float) * n);GET_TIME(t0);
        matvec_simple_listing5(n, vec_out, mat, vec_in);GET_TIME(t1);
        float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
        times[i] = el_t;
    }
    mean = Average(times, 10);
    sd = standardDeviation(times, 10);
    req_n = requiredSampleSize(sd, mean);
    printf("Average time : %f\n", mean);
    printf("Required sample size for 95 percent confidence : %d\n", req_n);
    if (req_n > 10) {
        float *times_2 = (float *) malloc(sizeof(float) * req_n);
        printf("Automatically running with required sample size");
        for (int i = 0; i < req_n; ++i) {
            memset(vec_out, 0, sizeof(float) * n);GET_TIME(t0);
            matvec_simple_listing5(n, vec_out, mat, vec_in);GET_TIME(t1);
            float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
            times_2[i] = el_t;
        }
        mean = Average(times_2, req_n);
        sd = standardDeviation(times_2, req_n);
        req_n = requiredSampleSize(sd, mean);
        printf("Average time : %f\n", mean);
        free(times_2);
    }
}

void driveMatVecCPU_listing6(const float *mat, const float *vec_in, float *vec_out, int n) {
    struct timespec t0, t1;
    unsigned long sec, nsec;
    float mean, sd;
    float times[10];
    int req_n;
    for (int i = 0; i < 10; ++i) {
        memset(vec_out, 0, sizeof(float) * n);GET_TIME(t0);
        matvec_unrolled_listing6(n, vec_out, mat, vec_in);GET_TIME(t1);
        float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
        times[i] = el_t;
    }
    mean = Average(times, 10);
    sd = standardDeviation(times, 10);
    req_n = requiredSampleSize(sd, mean);
    printf("Average time : %f\n", mean);
    printf("Required sample size for 95 percent confidence : %d\n", req_n);
    if (req_n > 10) {
        float *times_2 = (float *) malloc(sizeof(float) * req_n);
        printf("Automatically running with required sample size");
        for (int i = 0; i < req_n; ++i) {
            memset(vec_out, 0, sizeof(float) * n);GET_TIME(t0);
            matvec_unrolled_listing6(n, vec_out, mat, vec_in);GET_TIME(t1);
            float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
            times_2[i] = el_t;
        }
        mean = Average(times_2, req_n);
        sd = standardDeviation(times_2, req_n);
        req_n = requiredSampleSize(sd, mean);
        printf("Average time : %f\n", mean);
        free(times_2);
    }
}

void driveMatVecSSE(const float *mat, const float *vec_in, float *vec_out, int n) {
    struct timespec t0, t1;
    unsigned long sec, nsec;
    float mean, sd;
    int initialCount = 10;
    float times[initialCount];
    int req_n;
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
    sd = standardDeviation(times, initialCount);
    req_n = requiredSampleSize(sd, mean);
    printf("Average time : %f\n", mean);
    printf("Required sample size for 95 percent confidence : %d\n", req_n);
    if (req_n > 10) {
        float *times_2 = (float *) malloc(sizeof(float) * req_n);
        printf("Automatically running with required sample size");
        for (int i = 0; i < req_n; ++i) {
            memset(vec_out, 0, sizeof(float) * n);
            if (n >= 16) { GET_TIME(t0);
                matvec_unrolled_16sse(n, vec_out, mat, vec_in);GET_TIME(t1);
            } else { GET_TIME(t0);
                matvec_unrolled_sse_quite(n, vec_out, mat, vec_in);GET_TIME(t1);
            }
            float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
            times_2[i] = el_t;
        }
        mean = Average(times_2, req_n);
        sd = standardDeviation(times_2, req_n);
        req_n = requiredSampleSize(sd, mean);
        printf("Average time : %f\n", mean);
        free(times_2);
    }
}

void driveMatVecAuto(const float *mat, const float *vec_in, float *vec_out, int n) {
    struct timespec t0, t1;
    unsigned long sec, nsec;
    float mean, sd;
    float times[10];
    int req_n;
    for (int i = 0; i < 10; ++i) {
        memset(vec_out, 0, sizeof(float) * n);GET_TIME(t0);
        matvec_unrolled_auto(n, vec_out, mat, vec_in);GET_TIME(t1);
        float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
        times[i] = el_t;
    }
    mean = Average(times, 10);
    sd = standardDeviation(times, 10);
    req_n = requiredSampleSize(sd, mean);
    printf("Average time : %f\n", mean);
    printf("Required sample size for 95 percent confidence : %d\n", req_n);
    if (req_n > 10) {
        float *times_2 = (float *) malloc(sizeof(float) * req_n);
        printf("Automatically running with required sample size");
        for (int i = 0; i < req_n; ++i) {
            memset(vec_out, 0, sizeof(float) * n);GET_TIME(t0);
            matvec_unrolled_auto(n, vec_out, mat, vec_in);GET_TIME(t1);
            float el_t = elapsed_time_microsec(&t0, &t1, &sec, &nsec);
            times_2[i] = el_t;
        }
        mean = Average(times_2, req_n);
        sd = standardDeviation(times_2, req_n);
        req_n = requiredSampleSize(sd, mean);
        printf("Average time : %f\n", mean);
        free(times_2);
    }
}