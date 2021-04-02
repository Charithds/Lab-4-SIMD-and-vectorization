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

#define XMM_ALIGNMENT_BYTES 32
#define checkMem(mem) if(!mem){fprintf(stderr, "Memory allocation failed\n"),abort();}
#define COLUMNS 200

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
    getArguments(argc, argv, &n);

    matrixCreationNByN_1D(n, n, &mat0);
    checkMem(mat0)

    assert(!(n & 0x3) & !(cols & 0x3) && "Dimension of matrix and vectors should be multiple of 4");

    printf("Starting calculation...\n");
    printf("All the times are shown in micro seconds...\n");
		printf("Program will create %d x %d matrix and a %dx1 vector for calculations\n", n, n, n);
		// vector creation
		matrixCreationNByN_1D(n, 1, &in_vec);
		checkMem(in_vec)

		out_vec_simple = _mm_malloc(sizeof(float) * n, XMM_ALIGNMENT_BYTES);
		checkMem(out_vec_simple)
		printf("\nRunning listing 5 Program\n");
		driveMatVecCPU_listing5(n);

		out_vec_simple_list6 = _mm_malloc(sizeof(float) * n, XMM_ALIGNMENT_BYTES);
		checkMem(out_vec_simple_list6)
		printf("\nRunning listing 6 Program\n");
		driveMatVecCPU_listing6(n);

		out_vec_sse = _mm_malloc(sizeof(float) * n, XMM_ALIGNMENT_BYTES);
		checkMem(out_vec_sse)
		printf("\nRunning listing 6 sse version\n");
		driveMatVecSSE(mat0, in_vec, out_vec_sse, n);
		
		//printf("\nRunning listing 7 version\n");
		//driveMatMatCPU_listing7(n);

		_mm_free(in_vec);
		_mm_free(out_vec_simple);
		_mm_free(out_vec_simple_list6);
		_mm_free(out_vec_sse);
		
    _mm_free(mat0);
    mat0 = NULL;
    return 0;
}