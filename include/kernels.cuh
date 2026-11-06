#pragma once
#include "common.h"

// State vector initialization (Embarassingly Parallel)

__global__ void init_state_k(complex_t* state, const size_t N);

void init_state_with_kernel(complex_t* state, const size_t N);
void init_state_native(complex_t* state, const size_t N);

// Gate Application Kernels (Embarassingly Parallel)

__global__ void apply_u_gate_k(complex_t* state, const size_t pairs, const size_t target_qubit,
	const complex_t m00, const complex_t m01, const complex_t m10, const complex_t m11);
__global__ void apply_bi_gate_k(complex_t* state, const size_t quarts, const size_t target_qubit, const size_t control_qubit, matrix4x4 gate);
__global__ void apply_tri_gate_k(complex_t* state, const size_t octs, const size_t control_qubit_1, const size_t control_qubit_2, const size_t target_qubit, matrix8x8 gate);

void apply_u_gate(complex_t* state, const size_t N, const complex_t G[2][2], const size_t target_qubit);
void apply_bi_gate(complex_t* state, const size_t N, const complex_t G[4][4], const size_t target_qubit, const size_t control_qubit);
void apply_tri_gate(complex_t* state, const size_t N, const complex_t G[8][8], const size_t control_qubit_1, const size_t control_qubit_2, const size_t target_qubit);

// Calculate Probabilities (Embarassingly Parallel)

__global__ void sq_norm_k(complex_t* state, float *prob, const size_t N);

void calc_probabilities(complex_t* state, float *prob, const size_t N);

// Float Sum Reduction (Non-Embarassingly Parallel)

__global__ void sum2_k(float* prob, const size_t gap, const size_t N);
__global__ void sum16_k(float* prob, const size_t gap, const size_t N);
__global__ void sum16_lmem_k(float* prob, const size_t curr_N);

float sum2(float *prob, const size_t N);
float sum16(float *prob, const size_t N);
float sum16_lmem(float *prob, const size_t N);

// Measurement (Both)

__global__ void measure_collapse_16_lmem_k(const complex_t* state, float* temp_sums, const int target_qubit, const size_t N);
__global__ void measure_renormalize_k(complex_t* state, const int target_qubit, const int measured, const float norm_factor, const size_t N);

char measure(complex_t* state, const size_t N, const int target_qubit);
char measure_time_test(complex_t* state, const size_t N, const int target_qubit, float* collapse_ms, float* renorm_ms);