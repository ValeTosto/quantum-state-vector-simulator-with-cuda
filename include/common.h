#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thrust/complex.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define QBITS 28

#define LWS 256

using complex_t = thrust::complex<float>;

struct matrix4x4 {
	complex_t matrix[4][4];
};

struct matrix8x8 {
	complex_t m[8][8];
};

void check_cuda(cudaError_t err, const char* src, int line);
void verify_init(const complex_t* state, const size_t N);
void print_state(const complex_t* state, const size_t N);
void print_probs(const float* prob, const size_t N);

#define CHECK_CUDA(err) check_cuda(err, __FILE__, __LINE__)

