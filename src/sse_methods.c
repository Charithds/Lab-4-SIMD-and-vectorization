//
// Created by krv on 2/28/17.
//
#include <xmmintrin.h>
#include <smmintrin.h>
#include <stdio.h>
#include "sse_methods.h"
#include "util.h"

void matvec_unrolled_sse_quite(int n, float *vec_c, const float *mat_a, const float *vec_b) {
    // NOTE : Matrix and Vector both must have dimensions which are multiples of 4
    for (int i = 0; i < n; i += 1) {
        for (int j = 0; j < n; j += 4) {
            // load the vector
            __m128 x0 = _mm_load_ps(&vec_b[j]);
//            printf("\nvector :  ");
//            print_vector_ps(x0);

            // load the matirx mat_a current section to a v0-v1 four vectors
            __m128 v0 = _mm_load_ps(&mat_a[i * n + j]);
//            printf("\nmatrix row :  ");
//            print_vector_ps(v0);

            // dot product
            __m128 rslt = _mm_dp_ps(x0, v0, 241);
//            printf("\nresult :  ");
//            print_vector_ps(rslt);
            vec_c[i] += _mm_cvtss_f32(rslt);
        }
    }
}

void matvec_unrolled_16sse(int n, float *vec_c, const float *mat_a, const float *vec_b) {
    // NOTE : Matrix and Vector both must have dimensions which are multiples of 4
    int unroll16Size = n / 16;  // expect an integer division
    int unrolled_num = unroll16Size * 16;
    int rest = n - unrolled_num;

    for (int i = 0; i < n; i += 1) {
        int j = 0;
        for (int k = 0; k < unroll16Size; ++k) {
            for (; j < unrolled_num; j += 16) {

                // load next 4 floats from input vector
                __m128 x0 = _mm_load_ps(&vec_b[j]);
                // load next 4 floats from input vector
                __m128 x1 = _mm_load_ps(&vec_b[j + 4]);
                // load next 4 floats from input matrix
                __m128 v0 = _mm_load_ps(&mat_a[i * n + j]);
                // load next 4 floats from input matrix
                __m128 v1 = _mm_load_ps(&mat_a[i * n + j + 4]);

                // Dot product
                __m128 rslt_m0 = _mm_dp_ps(x0, v0, 241);
                __m128 rslt_m1 = _mm_dp_ps(x1, v1, 241);

                // load next 4 floats from input vector
                x0 = _mm_load_ps(&vec_b[j + 8]);
                // load next 4 floats from input vector
                x1 = _mm_load_ps(&vec_b[j + 12]);
                // load next 4 floats from input matrix
                v0 = _mm_load_ps(&mat_a[i * n + j + 8]);
                // load next 4 floats from input matrix
                v1 = _mm_load_ps(&mat_a[i * n + j + 12]);

                // Dot product
                __m128 rslt_m2 = _mm_dp_ps(x0, v0, 241);
                __m128 rslt_m3 = _mm_dp_ps(x1, v1, 241);

                vec_c[i] += _mm_cvtss_f32(rslt_m0);
                vec_c[i] += _mm_cvtss_f32(rslt_m1);
                vec_c[i] += _mm_cvtss_f32(rslt_m2);
                vec_c[i] += _mm_cvtss_f32(rslt_m3);

            }
        }
        if (rest > 0) {
            for (j = unrolled_num; j < n; j += 4) {

                __m128 x0 = _mm_load_ps(&vec_b[j]);
                __m128 v0 = _mm_load_ps(&mat_a[i * n + j]);

                // dot product
                __m128 rslt = _mm_dp_ps(x0, v0, 241);

                vec_c[i] += _mm_cvtss_f32(rslt);
            }
        }
    }
//    printVector(vec_c, n);
}

void matmat_listing7_sse(float *mat_c, int a_h, int n, const float *mat_a, int b_w, const float *mat_b) {
    for (int i = 0; i < a_h; i += 1) {
        for (int j = 0; j < n; j += 1) {
            __m128 x0 = _mm_set1_ps(mat_a[i * n + j]);
            for (int k = 0; k < b_w; k += 4) {
                __m128 temp = _mm_loadu_ps(&mat_c[i * b_w + k]);
                __m128 v0 = _mm_loadu_ps(&mat_b[j * b_w + k]);
                __m128 m0 = _mm_mul_ps(x0, v0);
                temp = _mm_add_ps(m0, temp);
                _mm_storeu_ps(&mat_c[i * b_w + k], temp);
            }
        }
    }
}