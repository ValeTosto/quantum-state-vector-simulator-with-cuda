#include "common.h"
#include "kernels.cuh"
#include "gates.h"
#include <time.h>


int main(int argc, char* argv[])
{

	if (argc != 3) {
		fprintf(stderr, "invalid arguments, correct usage: [QBITS] [TEST_MODE]\nQBITS: int in range [8-29], will only print the full vector if <=10;\nTEST_MODE: 0 = Full Hadamard Circuit (Expected uniform probability)\n1 = GHZ (Expected 0.5 probability on 00...00 and 11...11)\n2 = Single 0-1-2 Toffoli\n3 = Single last three qubits Toffoli\n4 = Test slow measure times\n5 = Test fast measure Times");
		exit(1);
	}

	int QBITS = atoi(argv[1]);
	int TEST_MODE = atoi(argv[2]);

	if (QBITS < 8 || QBITS > 29) {
		fprintf(stderr, "invalid qubits, must be in range [8-29]");
		exit(2);
	}

	const size_t N = 1ULL << QBITS;
	const size_t MEMSIZE_S = N * sizeof(complex_t);
	const size_t MEMSIZE_P = N * sizeof(float);

	// init buffer

	complex_t* h_state, * d_state;
	float *h_prob, *d_prob;
	
	CHECK_CUDA(cudaMalloc(&d_state, MEMSIZE_S));
	CHECK_CUDA(cudaMalloc(&d_prob, MEMSIZE_P));

	h_state = (complex_t*)malloc(MEMSIZE_S);
	if (!h_state) {
		fprintf(stderr, "h_state init failed");
		exit(3);
	}   

	h_prob = (float*)malloc(MEMSIZE_P);
	if (!h_prob) {
		fprintf(stderr, "h_prob init failed");
		exit(4);
	}

	cudaEvent_t preInit, preApply, postApply, postCopy, preSum, postSum, preMeasure, postMeasure;

	CHECK_CUDA(cudaEventCreate(&preInit));
	CHECK_CUDA(cudaEventCreate(&preApply));
	CHECK_CUDA(cudaEventCreate(&postApply));
	CHECK_CUDA(cudaEventCreate(&postCopy));
	CHECK_CUDA(cudaEventCreate(&preSum));
	CHECK_CUDA(cudaEventCreate(&postSum));
	CHECK_CUDA(cudaEventCreate(&preMeasure));
	CHECK_CUDA(cudaEventCreate(&postMeasure));

	// init state vector

	CHECK_CUDA(cudaEventRecord(preInit));

	init_state_native(d_state, N);

	CHECK_CUDA(cudaEventRecord(preApply));

	// apply selected circuit

	int gates_applied;

	if (TEST_MODE == 0 || TEST_MODE == 4|| TEST_MODE==5) {
		// all Hadamard: uniform distribution
		gates_applied = QBITS;
		for (int i = 0; i < QBITS; i++)
			apply_u_gate(d_state, N, Gates::H, i);
	}
	else if (TEST_MODE == 1) {
		// GHZ state
		gates_applied = QBITS;
		apply_u_gate(d_state, N, Gates::H, 0);
		for (int i = 1; i < QBITS; i++)
			apply_bi_gate(d_state, N, Gates::CNOT, i, i-1);
	}
	else if (TEST_MODE == 2) {
		// Slow Toffoli
		gates_applied = 1;
		apply_tri_gate(d_state, N, Gates::Toffoli, 0, 1, 2);
	}
	else if (TEST_MODE == 3) {
		//Fast Toffoli
		gates_applied = 1;
		apply_tri_gate(d_state, N, Gates::Toffoli, QBITS-3, QBITS-2, QBITS-1);
	}
	
	CHECK_CUDA(cudaEventRecord(postApply));

	// copy state vector 

	CHECK_CUDA(cudaMemcpy(h_state, d_state, MEMSIZE_S, cudaMemcpyDeviceToHost));

	CHECK_CUDA(cudaEventRecord(postCopy));

	// print state vector

	if (QBITS <= 10) {
		printf("\nState Vector:\n");
		print_state(h_state, N, QBITS);
	}

	// check sum equals 1

	calc_probabilities(d_state, d_prob, N);

	CHECK_CUDA(cudaMemcpy(h_prob, d_prob, MEMSIZE_P, cudaMemcpyDeviceToHost));

	if (QBITS <= 10) {
		printf("\nProbabilities Vector:\n");
		print_probs(h_prob, N, QBITS);
	}

	CHECK_CUDA(cudaEventRecord(preSum));

	float sum = sum16_lmem(d_prob, N);

	CHECK_CUDA(cudaEventRecord(postSum));

	printf("\nsum = %f\n", sum);

	// measure all qubits

	srand(time(NULL));

	for (int w = 0; w < 100; w++) {
		rand();
	}

	char* measured;

	float collapse_ms, renorm_ms;
	if (TEST_MODE == 4 || TEST_MODE==5) {
		int target = TEST_MODE == 5 ? QBITS - 1 : 0;
		measure_time_test(d_state, N, target , &collapse_ms, &renorm_ms);
	}
	else {
		measured = (char*)malloc(sizeof(char) * (QBITS + 1));

		CHECK_CUDA(cudaEventRecord(preMeasure));

		for (int i = 0; i < QBITS; i++)
			measured[i] = measure(d_state, N, i);

		CHECK_CUDA(cudaEventRecord(postMeasure));

		measured[QBITS] = '\0';

		printf("\nmeasure result: %s\n\n", measured);
		if (measured) free(measured);

		CHECK_CUDA(cudaEventSynchronize(postMeasure));
	}

	// get time stats

	float init_ms, apply_ms, copy_ms, sum_ms, measure_ms;

	CHECK_CUDA(cudaEventElapsedTime(&init_ms, preInit, preApply));
	CHECK_CUDA(cudaEventElapsedTime(&apply_ms, preApply, postApply));
	CHECK_CUDA(cudaEventElapsedTime(&copy_ms, postApply, postCopy));
	CHECK_CUDA(cudaEventElapsedTime(&sum_ms, preSum, postSum));
	

	printf("init: %.4g ms, %.4g GB/s, %.4g GE/s\n", init_ms, MEMSIZE_S / init_ms / 1.0e6, N / init_ms / 1.0e6);
	printf("gates: %.4g ms, %.4g GB/s, %.4g GE/s\n", apply_ms, 2 * gates_applied * MEMSIZE_S/apply_ms/1.0e6, 2 * gates_applied * N / apply_ms/1.0e6);
	printf("copy: %.4g ms, %.4g GB/s, %.4g GE/s\n", copy_ms, MEMSIZE_S / copy_ms / 1.0e6, N / copy_ms / 1.0e6);
	printf("sum: %.4g ms, %.4g GE/s\n", sum_ms, N / sum_ms / 1.0e6);
	
	if (TEST_MODE == 4 || TEST_MODE == 5) {
		printf("measure_collapse: %.4g ms, %.4g GE/s\n", collapse_ms, N / collapse_ms / 1.0e6);
		printf("measure_renorm: %.4g ms, %.4g GB/s, %.4g GE/s\n", renorm_ms, 1.5 * MEMSIZE_S / renorm_ms / 1.0e6, N / renorm_ms / 1.0e6);
	}
	else {
		CHECK_CUDA(cudaEventElapsedTime(&measure_ms, preMeasure, postMeasure));
		printf("measure: %.4g ms, %.4g GE/s\n", measure_ms, QBITS * N / measure_ms / 1.0e6);
	}

	// free memory
	if (h_state) free(h_state);
	if (h_prob) free(h_prob);
	if (d_state) cudaFree(d_state);
	if (d_prob) cudaFree(d_prob);

	CHECK_CUDA(cudaEventDestroy(preApply));
	CHECK_CUDA(cudaEventDestroy(preInit));
	CHECK_CUDA(cudaEventDestroy(preMeasure));
	CHECK_CUDA(cudaEventDestroy(preSum));
	CHECK_CUDA(cudaEventDestroy(postApply));
	CHECK_CUDA(cudaEventDestroy(postCopy));
	CHECK_CUDA(cudaEventDestroy(postMeasure));
	CHECK_CUDA(cudaEventDestroy(postSum));

	return 0;
}