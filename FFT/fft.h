//Do NOT MODIFY THIS FILE

#ifndef FFT_H
#define FFT_H

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gputimer.h"
#include "gpuerrors.h"

//#include "cuPrintf.cu"

#define PI 3.14159265


void gpuKernel_simple(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M);

void gpuKernel_efficient(float* x_r_d, float* x_i_d, float* X_r_d, float* X_i_d, const unsigned int N, const unsigned int M);


#endif

//Do NOT MODIFY THIS FILE