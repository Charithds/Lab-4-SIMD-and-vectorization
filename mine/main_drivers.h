#ifndef ACA_LAB3_DRIVERS_H
#define ACA_LAB3_DRIVERS_H

// Matrix X Vector versions
void driveMatVecCPU_listing5(int n);

void driveMatVecCPU_listing6(int n);

void driveMatMatCPU_listing7(int n);

void driveMatMatCPU_listing7_SSE(int n);

void driveMatVecSSE(const float *mat, const float *vec_in, float *vec_out, int n);

void driveMatVecAuto(const float *mat, const float *vec_in, float *vec_out, int n);

#endif //ACA_LAB3_DRIVERS_H
