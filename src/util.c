//
// Created by jawadhsr on 2/27/17.
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <xmmintrin.h>
#include <getopt.h>
#include "util.h"

// N = (100zs/rx)^2, r - accuracy (5%), s - standard deviation, x - Average, z = 1.960 (95%) - TAKE THE CEIL
/// This function gives the required number of samples to take to achieve
/// accuracy of 5% and 95% confidence
/// \param sd standard deviation
/// \param mean average
/// \return

long requiredSampleSize(float sd, float mean) {
    long N = (long) ceil(((float) 100 * 1.960 * sd) / (5 * mean));
    return N;
}

float Average(float *times, int numSamples) {
    float sum = 0;
    for (int i = 0; i < numSamples; ++i) {
        sum += times[i];
    }
    return (double) sum / numSamples;
}

float standardDeviation(float times[], int numSamples) {
    float u = Average(times, numSamples);
    float variance = 0;
    for (int i = 0; i < numSamples; ++i) {
        variance += pow(times[i] - u, 2);
    }
    variance = variance / numSamples;
    return sqrt(variance);
}

float elapsed_time_microsec(struct timespec *begin, struct timespec *end,
                            unsigned long *sec, unsigned long *nsec) {

    // make sure end time is after the begin
    assert(end->tv_sec > begin->tv_sec ||
           (end->tv_sec == begin->tv_sec &&
            end->tv_nsec >= begin->tv_nsec));

    if (end->tv_nsec < begin->tv_nsec) {
        *nsec = 1000000000 - (begin->tv_nsec - end->tv_nsec);
        *sec = end->tv_sec - begin->tv_sec - 1;
    } else {
        *nsec = end->tv_nsec - begin->tv_nsec;
        *sec = end->tv_sec - begin->tv_sec;
    }
    return (float) (*sec) * 1000 + ((float) (*nsec)) * 1E-6;
}

int getArguments(int argc, char *argv[], int *n, short *mat_vec_ver, short *mat_mat_ver, short *c_ver, short *sse_ver,
                 short *a_vec_ver, short *test, short *listing6) {
    int c;
    while ((c = getopt(argc, argv, "n:hvmcsat5")) != -1) {
        switch (c) {
            case 'n':
                *n = atoi(optarg);
                break;
            case 'v':
                *mat_vec_ver = 1;
                break;
            case 'c':
                *c_ver = 1;
                break;
            case 's':
                *sse_ver = 1;
                break;
            case 'a':
                *a_vec_ver = 1;
                break;
            case '5':
                *listing6 = 1;
                break;
            case 'm':
                *mat_mat_ver = 1;
                break;
            case 't':
                *test = 1;
                break;
            case 'h':
                printf("-m\t\t\t-\trun the matrix X matrix version.\n");
                printf("-v\t\t\t-\trun the matrix X vector version.\n");
                printf("-c\t\t\t-\trun the simple version.\n");
                printf("-s\t\t\t-\trun the sse version.\n");
                printf("-a\t\t\t-\trun the auto vectorized version.\n");
                printf("-t\t\t\t-\trun all versions and verify against simple version.\n");
                printf("-h\t\t\t-\tShow this menu\n");
                printf("example for the assignment.\n");
                printf("eg : ./aca_lab3 -n 100 -vc\n");
                printf("\t Run with matrix size 100x100(n), vector size 100(n), simple version(c) - matrix-vector (v)\n");
                printf("eg : ./aca_lab3 -n 10 -vs\n");
                printf("\t Run with matrix size 10x10(n), vector size 10(n), sse version(s) - matrix-vector (v)\n");
                printf("eg : ./aca_lab3 -n 100 -vt\n");
                printf("\t Run with matrix size 100x100(n), vector size 100(n), verification phase(t) - matrix-vector (v)\n");
                printf("eg : ./aca_lab3 -n 100 -mc\n");
                printf("\t Run with two matrices of size 100x100(n) simple version(c) - matrix-matrix (m)\n");
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