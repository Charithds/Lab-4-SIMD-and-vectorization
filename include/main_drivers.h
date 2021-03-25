//
// Created by krv on 3/10/17.
//

#ifndef ACA_LAB3_DRIVERS_H
#define ACA_LAB3_DRIVERS_H

// Matrix X Matrix versions
void driveMatMatCPU(const float *mat_a, const float *mat_b, float *mat_c, int n, int cols);

void driveMatMat_SSE(const float *mat_a, const float *mat_b, float *mat_c, int n, int cols);

void driveMatMatAuto(const float *mat_a, const float *mat_b, float *mat_c, int n, int cols);

// Matrix X Vector versions
void driveMatVecCPU_listing5(const float *mat, const float *vec_in, float *vec_out, int n);

void driveMatVecCPU_listing6(const float *mat, const float *vec_in, float *vec_out, int n);

void driveMatVecSSE(const float *mat, const float *vec_in, float *vec_out, int n);

void driveMatVecAuto(const float *mat, const float *vec_in, float *vec_out, int n);

#endif //ACA_LAB3_DRIVERS_H
