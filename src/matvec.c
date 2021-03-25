//
// Created by jawadhsr on 2/27/17.
//
#include "matvec.h"
#include "util.h"

/**
 * Listing 5
 * @param n
 * @param vec_c
 * @param mat_a
 * @param vec_b
 */
void matvec_simple_listing5(int n, float *vec_c,
                            const float *mat_a, const float *vec_b) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            vec_c[i] += mat_a[i * n + j] * vec_b[j];
}

/**
 * Listing 6
 * @param n
 * @param vec_c
 * @param mat_a
 * @param vec_b
 */
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