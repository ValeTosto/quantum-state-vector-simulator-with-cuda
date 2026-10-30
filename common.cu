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

void print_state(const complex_t* state, const size_t N)
{
	for (size_t i = 0; i < N; i++) {
		printf("state[%s] = (%f,%f) \n", std::bitset<QBITS>(i).to_string().c_str(), state[i].real(), state[i].imag());
	}
}

void print_probs(const float* prob, const size_t N)
{
	for (size_t i = 0; i < N; i++) {
		printf("state[%s] = %f \n", std::bitset<QBITS>(i).to_string().c_str(), prob[i]);
	}
}