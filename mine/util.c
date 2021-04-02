#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <xmmintrin.h>
#include <getopt.h>
#include "util.h"

double Average(double *times, int numSamples) {
    double sum = 0;
    for (int i = 0; i < numSamples; ++i) {
        sum += times[i];
    }
    return (double) sum / numSamples;
}

double elapsed_time(clock_t tic, clock_t toc) {
    return (double)(toc - tic) / CLOCKS_PER_SEC * 1000;
}

int getArguments(int argc, char *argv[], int *n) {
    int c;
    while ((c = getopt(argc, argv, "n:hvmcsat5")) != -1) {
        switch (c) {
            case 'n':
                *n = atoi(optarg);
                break;
            case '?':
                if (optopt == 'n') {
                    fprintf(stderr, "Option -n requires an integer point argument\n");
                } else {
                    fprintf(stderr, "Unknown option character\n");
                }
                return 1;
            default:
                abort();
        }
    }
    return 0;
}

void matrixCreationNByN_1D(int r, int c, float **mat_a) {
    *mat_a = _mm_malloc(sizeof(**mat_a) * r * c, XMM_ALIGNMENT_BYTES);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            (*mat_a)[i * c + j] = rand() / (float) RAND_MAX;
        }
    }
}