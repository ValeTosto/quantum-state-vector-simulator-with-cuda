#include "common.h"

#include <bitset>

void check_cuda(cudaError_t err, const char* src, int line)
{
	if (err != cudaSuccess) {
		fprintf(stderr, "%s:%d error %d = %s\n", src, line, err, cudaGetErrorString(err));
		exit(err);
	}
}

void verify_init(complex_t* state, const size_t N)
{
	if (state[0] != complex_t(1.0f, 0.0f)) {
		fprintf(stderr, "init failed: state[0] = (%f, %f), expected:(1.0, 0.0)",
			state[0].real(), state[0].imag());
		exit(1);
	}

	for (int i = 1; i < N; i++) {
		if (state[i] != complex_t(0.0f, 0.0f)) {
			fprintf(stderr, "init failed: state[%d] = (%f, %f), expected: (0.0, 0.0)",
				i, state[i].real(), state[i].imag());
			exit(1);
		}
	}
}

void idx_to_bin_str(size_t idx, const int QBITS, char* buffer) {
	for (int i = 0; i < QBITS; i++) {
		
		buffer[i] = ((idx >> (QBITS - 1 - i)) & 1) ? '1' : '0';
	}
	buffer[QBITS] = '\0'; 
}

void print_state(const complex_t* state, const size_t N, const int QBITS)
{
	char bin_str[32];
	for (size_t i = 0; i < N; i++) {
		idx_to_bin_str(i, QBITS, bin_str);
		printf("state[%s] = (%f,%f) \n", bin_str, state[i].real(), state[i].imag());
	}
}

void print_probs(const float* prob, const size_t N, const int QBITS)
{
	char bin_str[32];
	for (size_t i = 0; i < N; i++) {
		idx_to_bin_str(i, QBITS, bin_str);
		printf("state[%s] = %f \n", bin_str, prob[i]);
	}
}
