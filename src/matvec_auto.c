//
// Created by jawadhsr on 2/27/17.
//
#include "matvec_auto.h"

void matvec_unrolled_auto(int n, float *vec_c, const float *mat_a, const float *vec_b) {

#pragma omp parallel for simd
    for (int i = 0; i < n; i++) {
//        float sum =0;

//        #pragma GCC ivdep

        for (int j = 0; j < n; j += 4) {
            float f1 = mat_a[(i * n) + j] * vec_b[j];
            float f2 = mat_a[(i * n) + j + 1] * vec_b[j + 1];
            float f3 = mat_a[(i * n) + j + 2] * vec_b[j + 2];
            float f4 = mat_a[(i * n) + j + 3] * vec_b[j + 3];
//            vec_c[i] += f1;
//            answers[j] = mat_a[(i * n) + j] * vec_b[j];
            vec_c[i] += f1 + f2 + f3 + f4;
//            sum += f1+f2+f3+f4;

//            vec_c[i] += mat_a[(i * n) + j] * vec_b[j]
//                        + mat_a[(i * n) + j + 1] * vec_b[j + 1]
//                        + mat_a[(i * n) + j + 2] * vec_b[j + 2]
//                        + mat_a[(i * n) + j + 3] * vec_b[j + 3];
        }

//        for (int k = 0; k < n; ++k) {
//            vec_c[i] += answers[k];
//        }

    }
}