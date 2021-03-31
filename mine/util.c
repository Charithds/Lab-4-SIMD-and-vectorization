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
    return (double)(toc - tic) / CLOCKS_PER_SEC;
}

int getArguments(int argc, char *argv[], int *n, short *mat_vec_ver, short *mat_mat_ver, short *c_ver, short *sse_ver,
                 short *a_vec_ver, short *test, short *listing6) {
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
            (*mat_a)[i * c + j] = ((7 * i + j) & 0x0F) * 0x1P-2F;
        }
    }
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