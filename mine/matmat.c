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
#define COLUMNS 200


short mat_vec_ver = 0, mat_mat_ver = 0, c_ver = 0, sse_ver = 0, a_vec_ver = 0, test = 0, listing6;

int main(int argc, char *argv[]) {
	int n, cols = COLUMNS;
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