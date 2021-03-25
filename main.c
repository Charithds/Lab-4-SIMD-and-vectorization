/**
    Simple test programmes to test SSE instructions
 */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include "main_drivers.h"
#include "util.h"

//#define DEBUG

#ifdef DEBUG
#define COLUMNS 4
#else
#define COLUMNS 200
#endif

#define XMM_ALIGNMENT_BYTES 16

#define checkMem(mem) if(!mem){fprintf(stderr, "Memory allocation failed\n"),abort();}

static float *mat0 __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *mat1 __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *mat_ans_c __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *mat_ans_sse __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *mat_ans_auto __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *in_vec __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *out_vec_simple __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *out_vec_sse __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *out_vec_auto __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *out_vec_simple_list6 __attribute__((aligned (XMM_ALIGNMENT_BYTES)));

short mat_vec_ver = 0, mat_mat_ver = 0, c_ver = 0, sse_ver = 0, a_vec_ver = 0, test = 0, listing6;

int main(int argc, char *argv[]) {

    int n, cols = COLUMNS;
    time_t t;
    srand((unsigned) time(&t));
    getArguments(argc, argv, &n, &mat_vec_ver, &mat_mat_ver, &c_ver, &sse_ver, &a_vec_ver, &test, &listing6);

#ifdef DEBUG
    n = 8;
    t = 1;
    mat_mat_ver = 1;
    mat_vec_ver = 1;
#endif
    matrixCreationNByN_1D(n, n, &mat0);
    checkMem(mat0)

    assert(!(n & 0x3) & !(cols & 0x3) && "Dimension of matrix and vectors should be multiple of 4");

    printf("Starting calculation...\n");
    printf("All the times are shown in micro seconds...\n");
    if (mat_vec_ver) {
        printf("Program will create %d x %d matrix and a %dx1 vector for calculations\n", n, n, n);
        // vector creation
        matrixCreationNByN_1D(n, 1, &in_vec);
        checkMem(in_vec)

#ifdef DEBUG
        printf("Input Matrix\n");
        printNByCMat(mat0, n, n);
        printf("Input Vector\n");
        printVector(in_vec, n);
#endif
        // run 10 times get the average time
        if (c_ver || test) {
            out_vec_simple = _mm_malloc(sizeof(float) * n, XMM_ALIGNMENT_BYTES);
            checkMem(out_vec_simple)
            printf("\nRunning listing 5 C Program\n");
            driveMatVecCPU_listing5(mat0, in_vec, out_vec_simple, n);
        }
        if (listing6 || test) {
            out_vec_simple_list6 = _mm_malloc(sizeof(float) * n, XMM_ALIGNMENT_BYTES);
            checkMem(out_vec_simple_list6)
            printf("\nRunning listing 6 C Program\n");
            driveMatVecCPU_listing6(mat0, in_vec, out_vec_simple_list6, n);
        }
        if (sse_ver || test) {
            out_vec_sse = _mm_malloc(sizeof(float) * n, XMM_ALIGNMENT_BYTES);
            checkMem(out_vec_sse)
            printf("\nRunning sse version\n");
            driveMatVecSSE(mat0, in_vec, out_vec_sse, n);
        }
        if (a_vec_ver || test) {
            out_vec_auto = _mm_malloc(sizeof(float) * n, XMM_ALIGNMENT_BYTES);
            checkMem(out_vec_auto)
            printf("\nRunning auto vectorized version\n");
            driveMatVecAuto(mat0, in_vec, out_vec_auto, n);
        }

        _mm_free(in_vec);
        if (test) {
            printf("\nVerifying matrix vector multiplication\n");
            printf("Verifying sse (listing 6) and listing 5 C Program.\n");
            float error_sse_simple = 0;
            for (int i = 0; i < n; ++i) {
                error_sse_simple += fabsf(out_vec_simple[i] - out_vec_sse[i]);
            }

            if (error_sse_simple > FLT_EPSILON || isnanf(error_sse_simple)) {
                printf("\tSSE (listing 6) version verified against listing 5 C Program - NOT OK\n");
            } else {
                printf("\tSSE (listing 6) version verified against listing 5 C Program - OK\n");
            }
            printf("Verifying Listing 6 C and listing 5 C Program.\n");
            float error_list6_simple = 0;
            for (int i = 0; i < n; ++i) {
                error_list6_simple += fabsf(out_vec_simple_list6[i] - out_vec_simple[i]);
            }

            if (error_list6_simple > FLT_EPSILON || isnanf(error_list6_simple)) {
                printf("\tListing 6 C Program verified against listing 5 C Program - NOT OK\n");
            } else {
                printf("\tListing 6 C Program verified against listing 5 C Program - OK\n");
            }

            printf("Verifying auto and simple...\n");
            float error_auto_simple = 0;
            for (int i = 0; i < n; ++i) {
                error_auto_simple += fabsf(out_vec_auto[i] - out_vec_simple[i]);
            }
            if (error_auto_simple > FLT_EPSILON || isnanf(error_auto_simple)) {
                printf("\tAuto vectorized version verified against simple version - NOT OK\n");
            } else {
                printf("\tAuto vectorized version verified against simple version - OK\n");
            }
        }

        _mm_free(out_vec_simple);
        _mm_free(out_vec_simple_list6);
        _mm_free(out_vec_sse);
        _mm_free(out_vec_auto);
    }
    if (mat_mat_ver) {
        printf("Program will create random one %d x %d matrix and one %d x 200 matrix for calculations\n", n, n, n);
        matrixCreationNByN_1D(n, cols, &mat1);
        checkMem(mat1)
#ifdef DEBUG
        printf("Matrix A\n");
        printNByCMat(mat0, n, n);
        printf("Matrix B\n");
        printNByCMat(mat1, n, cols);
#endif
        if (c_ver || test) {
            mat_ans_c = _mm_malloc(sizeof(float) * n * cols, XMM_ALIGNMENT_BYTES);
            checkMem(mat_ans_c)
            printf("\nRunning mxm listing 7 C Program\n");
            driveMatMatCPU(mat0, mat1, mat_ans_c, n, cols);
#ifdef DEBUG
            printf("Matrix mat_ans_c\n");
            printNByCMat(mat_ans_c, n, cols);
#endif
        }

        if (sse_ver || test) {
            mat_ans_sse = _mm_malloc(sizeof(float) * n * cols, XMM_ALIGNMENT_BYTES);
            checkMem(mat_ans_sse)
            printf("\nRunning mxm listing 7 SSE version\n");
            driveMatMat_SSE(mat0, mat1, mat_ans_sse, n, cols);
#ifdef DEBUG
            printf("Matrix mat_ans_sse\n");
            printNByCMat(mat_ans_sse, n, cols);
#endif
        }

        if (a_vec_ver || test) {
            mat_ans_auto = _mm_malloc(sizeof(float) * n * cols, XMM_ALIGNMENT_BYTES);
            checkMem(mat_ans_auto)
            printf("\nRunning mxm listing 7 auto vectorized version\n");
            driveMatMatAuto(mat0, mat1, mat_ans_auto, n, cols);
        }

        if (test) {
            printf("\nVerifying matrix matrix multiplication\n");
            printf("Verifying listing 7 SSE and listing 7 C Program\n");
            float error_sse_simple = 0;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < cols; ++j) {
                    error_sse_simple += fabsf(mat_ans_c[i * cols + j] - mat_ans_sse[i * cols + j]);
                }
            }
            if (error_sse_simple > FLT_EPSILON || isnanf(error_sse_simple)) {
                printf("\tListing 7 SSE mxm version verified against listing 7 C Program - NOT OK\n");
            } else {
                printf("\tListing 7 SSE mxm version verified against listing 7 C Program - OK\n");
            }

            printf("Verifying listing 7 auto vectorized version and listing 7 C Program.\n");
            float error_auto_simple = 0;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < cols; ++j) {
                    error_auto_simple += fabsf(mat_ans_c[i * cols + j] - mat_ans_auto[i * cols + j]);
                }
            }

            if (error_auto_simple > FLT_EPSILON || isnanf(error_auto_simple)) {
                printf("\tListing 7 auto vectorized version verified against listing 7 C Program - NOT OK\n");
            } else {
                printf("\tListing 7 auto vectorized version verified against listing 7 C Program - OK\n");
            }
        }

        _mm_free(mat1);
        _mm_free(mat_ans_c);
        _mm_free(mat_ans_sse);
        _mm_free(mat_ans_auto);
    }

    _mm_free(mat0);
    mat0 = NULL;
    return 0;
}