#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <x86intrin.h>
// #include <xmmintrin.h>
// #include <smmintrin.h>
#include "util.h"
#include "main_drivers.h"

#define REPEATED_TIMES 1

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

    printNByCMat(mat_a, n, n);
    printf("\n");
    printVector(vec_b, n);
    printf("\n");
    // printNByCMat(&mat_a, n, n);

    for (int i = 0; i < n; i += 1) {
        int j = 0;
        for (int k = 0; k < unroll16Size; k++) {
            for (; j < unrolled_num; j += 16) {

                // load next 16 floats from input vector
                // printf("IM here\n");
                __m512 x = _mm512_load_ps(&vec_b[j]);
                // __m512 v = _mm512_load_ps(&vec_b[j]);
                // printf("IM here too\n");
                // printf("vars %d %d %d %d\n", i, n, j, &mat_a);
                __m512 v = _mm512_load_ps(&mat_a[i * n + j]);
                printf("IM here three\n");
                __m512 xv = _mm512_mul_ps(x, v);
                printf("IM here 4\n");
                float result = _mm512_mask_reduce_add_ps(0xFF, xv);
                printf("IM here 5\n");
                vec_c[i] += result;
                printf("%f\n", result);

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


void printNByCMat(const float *mat, int n, int c) {
    if (mat != NULL) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < c; ++i) {
                printf("%3.3f  ", mat[j * n + i]);
            }
            printf("\n");
        }
    }
}

void printVector(const float *vec, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%3.3f  ", vec[i]);
    }
    printf("\n");
}

void print_vector_ps(__m128 v) {
    const float *sv = (float *) &v;

    printf("%f %f %f %f\n",
           sv[0], sv[1], sv[2], sv[3]);
}