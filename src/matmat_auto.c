//
// Created by krv on 3/10/17.
//
#include "matmat_auto.h"

void matmat_auto(float *mat_c, int a_h, int com_n, const float *mat_a, int b_w, const float *mat_b) {
#pragma GCC ivdep
    for (int i = 0; i < a_h; i++) {
#pragma GCC ivdep
        for (int j = 0; j < b_w; j++) {
#pragma GCC ivdep
            for (int k = 0; k < com_n; k++) {
                mat_c[i * b_w + j] += mat_a[i * com_n + k] * mat_b[k * b_w + j];
            }
        }
    }
}