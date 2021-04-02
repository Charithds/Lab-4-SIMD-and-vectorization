#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include "util.h"

#define XMM_ALIGNMENT_BYTES 32
#define REPEATED_TIMES 200

static float *mat0 __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *mat1 __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float *mat_ans_c __attribute__((aligned (XMM_ALIGNMENT_BYTES)));

short mat_vec_ver = 0, mat_mat_ver = 0, c_ver = 0, sse_ver = 0, a_vec_ver = 0, test = 0, listing6;

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

int main(int argc, char *argv[]) {
	int n;
    time_t t;
    srand((unsigned) time(&t));
    getArguments(argc, argv, &n);

    printf("Starting calculation...\n");
    printf("All the times are shown in micro seconds...\n");
		printf("Program will create %d x %d matrix and a %dx%d vector for calculations\n", n, n, n, n);
		
		printf("\nRunning listing 7 Program\n");
		driveMatMatCPU_listing7(n);

		printf("\nRunning listing 7 SSE Program\n");
		driveMatMatCPU_listing7_SSE(n);
    return 0;
}