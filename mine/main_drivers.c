#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <x86intrin.h>
#include "util.h"
#include "main_drivers.h"

#define REPEATED_TIMES 200

static float *mat0 __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *mat1 __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *in_vec __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *vec_out __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *mat_ans_c __attribute__((aligned (XMM_ALIGNMENT_BYTES)));

// Matrix Vector Drivers
void matvec_simple_listing5(int n, float *vec_c, const float *mat_a, const float *vec_b) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            vec_c[i] += mat_a[i *n + j] * vec_b[j];
        }
    }
}

void driveMatVecCPU_listing5(int n) {
    double mean;
    double times[REPEATED_TIMES];

    vec_out = _mm_malloc(sizeof(float) * n, XMM_ALIGNMENT_BYTES);

    for (int i = 0; i < REPEATED_TIMES; ++i) {
        memset(vec_out, 0, sizeof(float) * n);
        matrixCreationNByN_1D(n, n, &mat0);
        matrixCreationNByN_1D(n, 1, &in_vec);
        clock_t tic = clock();
        matvec_simple_listing5(n, vec_out, mat0, in_vec);
        clock_t toc = clock();
        double el_t = elapsed_time(tic, toc);
        times[i] = el_t;
        _mm_free(in_vec);
        _mm_free(mat0);
    }
    mean = Average(times, REPEATED_TIMES);
    printf("Average time : %f\n", mean);
    _mm_free(vec_out);
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

void driveMatVecCPU_listing6(int n) {
    double mean;
    double times[REPEATED_TIMES];

    vec_out = _mm_malloc(sizeof(float) * n, XMM_ALIGNMENT_BYTES);

    for (int i = 0; i < REPEATED_TIMES; ++i) {
        memset(vec_out, 0, sizeof(float) * n);
        matrixCreationNByN_1D(n, n, &mat0);
        matrixCreationNByN_1D(n, 1, &in_vec);
        clock_t tic = clock();
        matvec_unrolled_listing6(n, vec_out, mat0, in_vec);
        clock_t toc = clock();
        double el_t = elapsed_time(tic, toc);
        times[i] = el_t;
        _mm_free(in_vec);
        _mm_free(mat0);
    }
    mean = Average(times, REPEATED_TIMES);
    printf("Average time : %f\n", mean);
    _mm_free(vec_out);
}

// listing 6 sse
void matvec_unrolled_16sse(int n, float *vec_c, const float *mat_a, const float *vec_b) {
    // NOTE : Matrix and Vector both must have dimensions which are multiples of 4
    int unroll16Size = n / 16;  // expect an integer division
    int unrolled_num = unroll16Size * 16;
    int rest = n - unrolled_num;

    for (int i = 0; i < n; i+=1) {
        vec_c[i] = 0.0;
        int j = 0;

        for (int k = 0; k < unroll16Size; ++k) {
            for (; j < unrolled_num; j += 4) {

                // load next 4 floats from input vector
                __m128 x0 = _mm_load_ps(&vec_b[j]);
                __m128 v0 = _mm_load_ps(&mat_a[i * n + j]);

                // Dot product
                __m128 rslt_m0 = _mm_dp_ps(x0, v0,0xFF);

                vec_c[i] += _mm_cvtss_f32(rslt_m0);
            }
        }
        if (rest > 0) {
            int mask = (1 << rest) - 1;
            for (j = unrolled_num; j < n; j += 4) {
                __m128 x0 = _mm_load_ps(&vec_b[j]);
                __m128 v0 = _mm_load_ps(&mat_a[i * n + j]);
                __m128 rslt = _mm_dp_ps(x0, v0, 0xFF);
                vec_c[i] += _mm_cvtss_f32(rslt);
            }
        }
    }
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

void matmat_listing7(int n, float *mat_c, const float *mat_a, const float *mat_b) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                mat_c[i * n + j] += mat_a[i * n + k] * mat_b[k * n + j];
            }
        }
    }
}

void driveMatMatCPU_listing7(int n) {
    double mean;
    double times[REPEATED_TIMES];

    for (int i = 0; i < REPEATED_TIMES; ++i) {
        matrixCreationNByN_1D(n, n, &mat0);
		matrixCreationNByN_1D(n, n, &mat1);
		matrixCreationNByN_1D(n, n, &mat_ans_c);
        memset(mat_ans_c, 0, sizeof(float) * n *n);
        clock_t tic = clock();
        matmat_listing7(n, mat_ans_c, mat0, mat1);
        clock_t toc = clock();
        double el_t = elapsed_time(tic, toc);
        times[i] = el_t;
    }
    mean = Average(times, REPEATED_TIMES);
    printf("Average time : %f\n", mean);
    _mm_free(mat0);
    _mm_free(mat1);
    _mm_free(mat_ans_c);
}

void matmat_listing7_SSE(int n, float *mat_c, const float *mat_a, const float *mat_b) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j+=4) {
            __m128 vR = _mm_setzero_ps();
            for (int k = 0; k < n; k++) {
                __m128 vA = _mm_set1_ps(mat_a[i * n + k]);  // load+broadcast is much cheaper than MOVD + 3 inserts (or especially 4x insert, which your new code is doing)
                __m128 vB = _mm_loadu_ps((__m128i*)&mat_b[k * n + j]);  // mat2[k][j+0..3]
                vR = _mm_add_ps(vR, _mm_mul_ps(vA, vB));
                mat_c[i * n + j] += mat_a[i * n + k] * mat_b[k * n + j];
            }
            _mm_storeu_ps((__m128*)&mat_c[i * n+ j], vR);
        }
    }
}

void driveMatMatCPU_listing7_SSE(int n) {
    double mean;
    double times[REPEATED_TIMES];
    
    float *mat0 __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
    float *mat1 __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
    float *mat_ans_c __attribute__((aligned (XMM_ALIGNMENT_BYTES)));

    for (int i = 0; i < REPEATED_TIMES; ++i) {
        matrixCreationNByN_1D(n, n, &mat0);
		matrixCreationNByN_1D(n, n, &mat1);
		matrixCreationNByN_1D(n, n, &mat_ans_c);
        memset(mat_ans_c, 0, sizeof(float) * n *n);
        clock_t tic = clock();
        matmat_listing7_SSE(n, mat_ans_c, mat0, mat1);
        clock_t toc = clock();
        double el_t = elapsed_time(tic, toc);
        times[i] = el_t;
    }
    mean = Average(times, REPEATED_TIMES);
    printf("Average time : %f\n", mean);
    _mm_free(mat0);
    _mm_free(mat1);
    _mm_free(mat_ans_c);
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