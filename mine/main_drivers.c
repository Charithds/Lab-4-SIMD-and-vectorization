#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <x86intrin.h>
// #include <xmmintrin.h>
// #include <smmintrin.h>
#include "util.h"
#include "main_drivers.h"

#define REPEATED_TIMES 200

// Matrix Vector Drivers
void matvec_simple_listing5(int n, float *vec_c,
                            const float *mat_a, const float *vec_b) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            vec_c[i] += mat_a[i * n + j] * vec_b[j];
}

void driveMatVecCPU_listing5(const float *mat, const float *vec_in, float *vec_out, int n) {
    double mean;
    double times[REPEATED_TIMES];
    for (int i = 0; i < REPEATED_TIMES; ++i) {
        memset(vec_out, 0, sizeof(float) * n);
        clock_t tic = clock();
        matvec_simple_listing5(n, vec_out, mat, vec_in);
        clock_t toc = clock();
        double el_t = elapsed_time(tic, toc);
        times[i] = el_t;
    }
    mean = Average(times, REPEATED_TIMES);
    printf("Average time : %f\n", mean);
}

// listing 6
void matvec_unrolled_listing6(int n, float *vec_c, const float *mat_a, const float *vec_b) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j += 4) {
            vec_c[i] += mat_a[i * n + j] * vec_b[j]
                        + mat_a[i * n + j + 1] * vec_b[j + 1]
                        + mat_a[i * n + j + 2] * vec_b[j + 2]
                        + mat_a[i * n + j + 3] * vec_b[j + 3];
        }
    }
}

void driveMatVecCPU_listing6(const float *mat, const float *vec_in, float *vec_out, int n) {
    double mean;
    double times[REPEATED_TIMES];
    for (int i = 0; i < REPEATED_TIMES; ++i) {
        memset(vec_out, 0, sizeof(float) * n);
        clock_t tic = clock();
        matvec_unrolled_listing6(n, vec_out, mat, vec_in);
        clock_t toc = clock();
        double el_t = elapsed_time(tic, toc);
        times[i] = el_t;
    }
    mean = Average(times, REPEATED_TIMES);
    printf("Average time : %f\n", mean);
}

// listing 6 sse
void matvec_unrolled_16sse(int n, float *vec_c, const float *mat_a, const float *vec_b) {
    // NOTE : Matrix and Vector both must have dimensions which are multiples of 4
    int unroll16Size = n / 16;  // expect an integer division
    int unrolled_num = unroll16Size * 16;
    int rest = n - unrolled_num;

    for (int i = 0; i < n; i += 1) {
        int j = 0;
        for (int k = 0; k < unroll16Size; ++k) {
            for (; j < unrolled_num; j += 16) {

                // load next 16 floats from input vector
                __m512 x = _mm512_load_ps(&vec_b[j]);
                // __m512 v = _mm512_load_ps(&vec_b[j]);
                __m512 v = _mm512_load_ps(&mat_a[i * n + j]);
                __m512 rslt = _mm512_mul_ps(x, v);

                 vec_c[i] += _mm512_cvtss_f32(rslt);
            }
        }
        // if (rest > 0) {
        //     for (j = unrolled_num; j < n; j += 4) {

        //         __m128 x0 = _mm_load_ps(&vec_b[j]);
        //         __m128 v0 = _mm_load_ps(&mat_a[i * n + j]);

        //         // dot product
        //         __m128 rslt = _mm_dp_ps(x0, v0, 0xff);

        //         vec_c[i] += _mm_cvtss_f32(rslt);
        //     }
        // }
    }
//    printVector(vec_c, n);
}

void driveMatVecSSE(const float *mat, const float *vec_in, float *vec_out, int n) {
    double mean;
    double times[REPEATED_TIMES];
    for (int i = 0; i < REPEATED_TIMES; ++i) {
        memset(vec_out, 0, sizeof(float) * n);
        clock_t tic = clock();
        matvec_unrolled_16sse(n, vec_out, mat, vec_in);
        clock_t toc = clock();
        double el_t = elapsed_time(tic, toc);
        times[i] = el_t;
    }
    mean = Average(times, REPEATED_TIMES);
    printf("Average time : %f\n", mean);
}