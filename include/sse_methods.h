//
// Created by krv on 2/28/17.
//

#ifndef ACA_LAB3_SSE_METHODS_H
#define ACA_LAB3_SSE_METHODS_H

#if !defined(__SSE3__) && !defined(__SSE4_1__)
#error This example requires SSE3 and SSE4.1
#endif


void matvec_unrolled_sse_quite(int n, float *vec_c, const float *mat_a, const float *vec_b);

void matvec_unrolled_16sse(int n, float *vec_c, const float *mat_a, const float *vec_b);

void matmat_listing7_sse(float *mat_c, int a_h, int com_n, const float *mat_a, int b_w, const float *mat_b);

#endif //ACA_LAB3_SSE_METHODS_H
