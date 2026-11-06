#include "kernels.cuh"

__global__ void init_state_k(complex_t* state, const size_t N)
{
	size_t i = threadIdx.x + (size_t)blockDim.x * blockIdx.x;
	if (i == 0)
		state[i] = complex_t(1.0f, 0.0f);
	else if (i < N) {
		state[i] = complex_t(0.0f, 0.0f);
	}
}

__global__ void init_zero_state_k(complex_t* state) {
	state[0] = complex_t(1.0, 0.0f);
}

__global__ void apply_u_gate_k(complex_t* state, const size_t pairs, const size_t target_qubit,
	const complex_t m00, const complex_t m01, const complex_t m10, const complex_t m11)
{
	// Applica al qubit [target_qubit] all'interno dello State Vector [state] il gate unario
	// [ m00  m01 ]
	// [ m10  m11 ]
	size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= pairs) return;

	size_t jump = 2 << target_qubit;
	size_t block_dim = 2 * jump;

	size_t block_index = i / jump;

	size_t offset = i % jump;

	size_t p_0 = block_dim * block_index + offset;
	size_t p_1 = p_0 + jump;

	// p_0 e p_1 sono i due indici dello state vector su cui il thread lavora.
	// Cio' che deve fare ora e' moltiplicare
	// [ state[p_0] ] [ m00  m01 ]
	// [ state[p_1] ] [ m10  m11 ]

	complex_t v0 = state[p_0];
	complex_t v1 = state[p_1];

	complex_t new_v0 = v0 * m00 + v1 * m01;
	complex_t new_v1 = v0 * m10 + v1 * m11;

	state[p_0] = new_v0;
	state[p_1] = new_v1;
}

__global__ void apply_u_gate_opt_k(complex_t* state, const size_t pairs, const size_t target_qubit,
	const complex_t m00, const complex_t m01, const complex_t m10, const complex_t m11)
{
	// Applica al qubit [target_qubit] all'interno dello State Vector [state] il gate unario
	// [ m00  m01 ]
	// [ m10  m11 ]
	size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= pairs) return;

	size_t mask = (1ULL << target_qubit) - 1;

	size_t lower = i & mask;

	size_t upper = (i >> target_qubit) << (target_qubit + 1);

	size_t p_0 = upper | lower;
	size_t p_1 = p_0 | (1ULL << target_qubit);

	complex_t v0 = state[p_0];
	complex_t v1 = state[p_1];

	// p_0 e p_1 sono i due indici dello state vector su cui il thread lavora.
	// Cio' che deve fare ora e' moltiplicare
	// [ v0 ] [ m00  m01 ]
	// [ v1 ] [ m10  m11 ]

	complex_t new_v0 = v0 * m00 + v1 * m01;
	complex_t new_v1 = v0 * m10 + v1 * m11;

	state[p_0] = new_v0;
	state[p_1] = new_v1;
}

__global__ void apply_bi_gate_opt_k(complex_t* state, const size_t quarts, const size_t target_qubit, const size_t control_qubit, matrix4x4 gate) {
	size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= quarts) return;

	size_t qubit_low = min(target_qubit, control_qubit);
	size_t qubit_high = max(target_qubit, control_qubit);

	size_t mask_low = (1ULL << qubit_low) - 1;
	size_t mask_high = (1ULL << qubit_high) - 1;
	size_t _00_idx = (i & mask_low) | ((i & ~mask_low) << 1);
	_00_idx = (_00_idx & mask_high) | ((_00_idx & ~mask_high) << 1);

	size_t _01_idx = _00_idx | (1ULL << target_qubit);
	size_t _10_idx = _00_idx | (1ULL << control_qubit);
	size_t _11_idx = _00_idx | (1ULL << control_qubit) | (1ULL << target_qubit);

	complex_t v00 = state[_00_idx];
	complex_t v01 = state[_01_idx];
	complex_t v10 = state[_10_idx];
	complex_t v11 = state[_11_idx];

	complex_t new_v00 = v00 * gate.matrix[0][0] + v01 * gate.matrix[0][1] + v10 * gate.matrix[0][2] + v11 * gate.matrix[0][3];
	complex_t new_v01 = v00 * gate.matrix[1][0] + v01 * gate.matrix[1][1] + v10 * gate.matrix[1][2] + v11 * gate.matrix[1][3];
	complex_t new_v10 = v00 * gate.matrix[2][0] + v01 * gate.matrix[2][1] + v10 * gate.matrix[2][2] + v11 * gate.matrix[2][3];
	complex_t new_v11 = v00 * gate.matrix[3][0] + v01 * gate.matrix[3][1] + v10 * gate.matrix[3][2] + v11 * gate.matrix[3][3];

	state[_00_idx] = new_v00;
	state[_01_idx] = new_v01;
	state[_10_idx] = new_v10;
	state[_11_idx] = new_v11;
}

__global__ void apply_tri_gate_k(complex_t* state, const size_t octs, const size_t control_qubit_1, const size_t control_qubit_2, const size_t target_qubit, matrix8x8 gate)
{
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= octs) return;

	size_t qubit_low = min(target_qubit, min(control_qubit_1, control_qubit_2));
	size_t qubit_high = max(target_qubit, max(control_qubit_1, control_qubit_2));
	size_t qubit_mid = control_qubit_1 + control_qubit_2 + target_qubit - qubit_low - qubit_high;

	size_t mask_low = (1ULL << qubit_low) - 1;
	size_t mask_mid = (1ULL << qubit_mid) - 1;
	size_t mask_high = (1ULL << qubit_high) - 1;
	size_t _000_idx = (i & mask_low) | ((i & ~mask_low) << 1);
	_000_idx = (_000_idx & mask_mid) | ((_000_idx & ~mask_mid) << 1);
	_000_idx = (_000_idx & mask_high) | ((_000_idx & ~mask_high) << 1);

	size_t _001_idx = _000_idx | (1ULL << target_qubit);
	size_t _010_idx = _000_idx | (1ULL << control_qubit_2);
	size_t _011_idx = _000_idx | (1ULL << control_qubit_2) | (1ULL << target_qubit);
	size_t _100_idx = _000_idx | (1ULL << control_qubit_1);
	size_t _101_idx = _000_idx | (1ULL << control_qubit_1) | (1ULL << target_qubit);
	size_t _110_idx = _000_idx | (1ULL << control_qubit_1) | (1ULL << control_qubit_2);
	size_t _111_idx = _000_idx | (1ULL << control_qubit_1) | (1ULL << control_qubit_2) | (1ULL << target_qubit);

	complex_t v000 = state[_000_idx];
	complex_t v001 = state[_001_idx];
	complex_t v010 = state[_010_idx];
	complex_t v011 = state[_011_idx];
	complex_t v100 = state[_100_idx];
	complex_t v101 = state[_101_idx];
	complex_t v110 = state[_110_idx];
	complex_t v111 = state[_111_idx];

	complex_t new_v000 = v000 * gate.m[0][0] + v001 * gate.m[0][1] + v010 * gate.m[0][2] + v011 * gate.m[0][3] + v100 * gate.m[0][4] + v101 * gate.m[0][5] + v110 * gate.m[0][6] + v111 * gate.m[0][7];
	complex_t new_v001 = v000 * gate.m[1][0] + v001 * gate.m[1][1] + v010 * gate.m[1][2] + v011 * gate.m[1][3] + v100 * gate.m[1][4] + v101 * gate.m[1][5] + v110 * gate.m[1][6] + v111 * gate.m[1][7];
	complex_t new_v010 = v000 * gate.m[2][0] + v001 * gate.m[2][1] + v010 * gate.m[2][2] + v011 * gate.m[2][3] + v100 * gate.m[2][4] + v101 * gate.m[2][5] + v110 * gate.m[2][6] + v111 * gate.m[2][7];
	complex_t new_v011 = v000 * gate.m[3][0] + v001 * gate.m[3][1] + v010 * gate.m[3][2] + v011 * gate.m[3][3] + v100 * gate.m[3][4] + v101 * gate.m[3][5] + v110 * gate.m[3][6] + v111 * gate.m[3][7];
	complex_t new_v100 = v000 * gate.m[4][0] + v001 * gate.m[4][1] + v010 * gate.m[4][2] + v011 * gate.m[4][3] + v100 * gate.m[4][4] + v101 * gate.m[4][5] + v110 * gate.m[4][6] + v111 * gate.m[4][7];
	complex_t new_v101 = v000 * gate.m[5][0] + v001 * gate.m[5][1] + v010 * gate.m[5][2] + v011 * gate.m[5][3] + v100 * gate.m[5][4] + v101 * gate.m[5][5] + v110 * gate.m[5][6] + v111 * gate.m[5][7];
	complex_t new_v110 = v000 * gate.m[6][0] + v001 * gate.m[6][1] + v010 * gate.m[6][2] + v011 * gate.m[6][3] + v100 * gate.m[6][4] + v101 * gate.m[6][5] + v110 * gate.m[6][6] + v111 * gate.m[6][7];
	complex_t new_v111 = v000 * gate.m[7][0] + v001 * gate.m[7][1] + v010 * gate.m[7][2] + v011 * gate.m[7][3] + v100 * gate.m[7][4] + v101 * gate.m[7][5] + v110 * gate.m[7][6] + v111 * gate.m[7][7];

	state[_000_idx] = new_v000;
	state[_001_idx] = new_v001;
	state[_010_idx] = new_v010;
	state[_011_idx] = new_v011;
	state[_100_idx] = new_v100;
	state[_101_idx] = new_v101;
	state[_110_idx] = new_v110;
	state[_111_idx] = new_v111;
}

__global__ void sq_norm_k(complex_t* state, float* prob, const size_t N)
{
	size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= N) return;

	complex_t c = state[i];
	prob[i] = c.real() * c.real() + c.imag() * c.imag();
}

__global__ void sum2_k(float* prob, const size_t gap, const size_t N)
{
	size_t i = 2*gap*(blockDim.x * blockIdx.x + threadIdx.x);
	if (i + gap < N) {
		prob[i] += prob[i+gap];
	}
}

__global__ void sum16_k(float* prob, const size_t gap, const size_t N)
{
	size_t i = 16 * gap * (blockDim.x * blockIdx.x + threadIdx.x);
	if (i + gap < N) {
		float sum = prob[i];
		for (int j = 1; j < 16; j++) {
			size_t idx = i + j * gap;
			if (idx < N)
				sum += prob[idx];
		}
		prob[i] = sum;
	}
}

__global__ void sum16_lmem_k(float* prob, const size_t currN)
{
	extern __shared__ float lmem[];

	size_t start = (size_t)blockDim.x * blockIdx.x * 16;
	size_t idx;

	float sum = 0;

	for (int j = 0; j < 16; j++) {
		idx = start + j * blockDim.x + threadIdx.x;
		if(idx < currN)
			sum += prob[idx];
	}

	lmem[threadIdx.x] = sum;
	
	for (int j = blockDim.x / 2; j > 0; j /= 2) {
		__syncthreads();
		if (threadIdx.x < j) {
			lmem[threadIdx.x] += lmem[threadIdx.x + j];
		}
	}

	if (threadIdx.x == 0) {
		prob[blockIdx.x] = lmem[0];
	}

}

__global__ void measure_collapse_16_lmem_k(const complex_t* state, float* temp_sums, const int target_qubit, const size_t npairs)
{
	extern __shared__ float lmem[];

	size_t start = (size_t)blockDim.x * blockIdx.x * 16;

	float sum = 0;
	size_t i;

	for (int j = 0; j < 16; j++) {
		i = start + j * blockDim.x + threadIdx.x;

		if (i < N) {
			if (((i >> target_qubit) & 1) == 0) {
				float r = state[i].real();
				float im = state[i].imag();
				sum += r * r + im * im;
			}
		}
	}

	lmem[threadIdx.x] = sum;

	for (int j = blockDim.x / 2; j > 0; j /= 2) {
		__syncthreads();
		if (threadIdx.x < j) {
			lmem[threadIdx.x] += lmem[threadIdx.x + j];
		}
	}

	if (threadIdx.x == 0) {
		temp_sums[blockIdx.x] = lmem[0];
	}
}

__global__ void measure_renormalize_k(complex_t* state, const int target_qubit, const int measured, const float norm_factor, const size_t N)
{
	size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;

	if (i < N) {
		int target_bit = (i >> target_qubit) & 1;

		if (target_bit == measured) {
			float r = state[i].real() * norm_factor;
			float im = state[i].imag() * norm_factor;
			state[i] = complex_t(r, im);
		}
		else {
			state[i] = complex_t(0.0f, 0.0f);
		}
	}
}

void init_state_with_kernel(complex_t* state, const size_t N)
{
	size_t blockDim = (N + LWS - 1) / LWS;
	init_state_k << <blockDim, LWS >> > (state, N);
}

void init_state_native(complex_t *state, const size_t N)
{
	CHECK_CUDA(cudaMemset(state, 0, N * sizeof(complex_t)));
	complex_t c1 = complex_t(1.0, 0.0);
	CHECK_CUDA(cudaMemcpy(state, &c1, sizeof(complex_t), cudaMemcpyHostToDevice));
}

void apply_u_gate(complex_t* state, const size_t N, const complex_t G[2][2], const size_t target_qubit)
{
	size_t blockNum = (N / 2 + LWS - 1) / LWS;
	apply_u_gate_opt_k << < blockNum, LWS >> > (state, N / 2, target_qubit,
		G[0][0], G[0][1], G[1][0], G[1][1]);
}

void apply_bi_gate(complex_t* state, const size_t N, const complex_t G[4][4], const size_t target_qubit, const size_t control_qubit)
{
	size_t blockNum = (N / 4 + LWS - 1) / LWS;
	matrix4x4 gate;
	memcpy(gate.matrix, G, sizeof(complex_t) * 4 * 4);

	apply_bi_gate_opt_k << < blockNum, LWS >> > (state, N / 4, target_qubit, control_qubit, gate);
}

void apply_tri_gate(complex_t* state, const size_t N, const complex_t G[8][8], const size_t control_qubit_1, const size_t control_qubit_2, const size_t target_qubit)
{
	const size_t blockNum = (N / 8 + LWS - 1) / LWS;
	matrix8x8 gate;
	memcpy(gate.m, G, sizeof(complex_t)*8*8);
	apply_tri_gate_k << <blockNum, LWS >> > (state, N / 8, control_qubit_1, control_qubit_2, target_qubit, gate);
}

void calc_probabilities(complex_t* state, float* prob, const size_t N)
{
	const size_t blockNum = (N + LWS - 1) / LWS;
	sq_norm_k << <blockNum, LWS >> > (state, prob, N);
}

float sum2(float* prob, const size_t N)

{
	for (size_t gap = 1; gap < N; gap *= 2) {
		size_t npairs = (N / gap) / 2;
		size_t blockNum = (npairs + LWS - 1) / LWS;
		
		sum2_k<<<blockNum, LWS>>>(prob, gap, N);
	}
	float ret;
	CHECK_CUDA(cudaMemcpy(&ret, &prob[0], sizeof(float), cudaMemcpyDeviceToHost));

	return ret;
}

float sum16(float* prob, const size_t N)

{
	for (size_t gap = 1; gap < N; gap *= 16) {
		size_t nhex = (N / gap) / 16;
		size_t blockNum = (nhex + LWS - 1) / LWS;

		sum16_k << <blockNum, LWS >> > (prob, gap, N);
	}
	float ret;
	CHECK_CUDA(cudaMemcpy(&ret, &prob[0], sizeof(float), cudaMemcpyDeviceToHost));

	return ret;
}

float sum16_lmem(float* prob, const size_t N) 
{
	size_t currN = N;
	size_t processed = LWS * 16;

	const size_t shared = LWS * sizeof(float);

	while (currN > 1) {

		size_t blockNum = (currN + processed - 1) / processed;

		sum16_lmem_k << <blockNum, LWS, shared >> > (prob, currN);

		currN = blockNum;
	}

	float ret;
	CHECK_CUDA(cudaMemcpy(&ret, &prob[0], sizeof(float), cudaMemcpyDeviceToHost));

	return ret;
}

char measure(complex_t* state, const size_t N, const int target_qubit)
{
	// measure p_0 for target qubit

	size_t processed = LWS * 16;
	size_t blockNum = (N + processed - 1) / processed;

	float* h_temp_sums, * d_temp_sums;
	CHECK_CUDA(cudaMalloc(&d_temp_sums, blockNum * sizeof(float)));

	h_temp_sums = (float*)malloc(blockNum * sizeof(float));
	if (!h_temp_sums) {
		fprintf(stderr, "h_temp_sums init failed\n");
		exit(4);
	}

	const size_t shared = LWS * sizeof(float);


	measure_collapse_16_lmem_k << < blockNum, LWS, shared >> > (state, d_temp_sums, target_qubit, N);

	CHECK_CUDA(cudaMemcpy(h_temp_sums, d_temp_sums, blockNum * sizeof(float), cudaMemcpyDeviceToHost));

	float p_0 = h_temp_sums[0];
	for (size_t i = 1; i < blockNum; i++)
		p_0 += h_temp_sums[i];

	float ran = (float)rand() / (float)RAND_MAX;
	int measured = (ran < p_0) ? 0 : 1;

	cudaFree(d_temp_sums);
	free(h_temp_sums);

	// renormalize state vector

	float p_measured = (measured == 0) ? p_0 : (1.0 - p_0);

	float norm_factor = 1 / sqrt(p_measured);

	blockNum = (N + LWS - 1) / LWS;

	measure_renormalize_k << <blockNum, LWS >> > (state, target_qubit, measured, norm_factor, N);


	return '0' + measured;

}

char measure_time_test(complex_t* state, const size_t N, const int target_qubit, float* collapse_ms, float* renorm_ms)
{
	// measure p_0 for target qubit

	size_t processed = LWS * 16;
	size_t blockNum = (N + processed - 1) / processed;

	float* h_temp_sums, *d_temp_sums;
	CHECK_CUDA( cudaMalloc(&d_temp_sums, blockNum * sizeof(float) ));

	h_temp_sums = (float*)malloc(blockNum * sizeof(float));
	if (!h_temp_sums) {
		fprintf(stderr, "h_temp_sums init failed\n");
		exit(4);
	}

	cudaEvent_t pre_collapse, post_collapse, pre_renorm, post_renorm;
	CHECK_CUDA(cudaEventCreate(&pre_collapse));
	CHECK_CUDA(cudaEventCreate(&post_collapse));
	CHECK_CUDA(cudaEventCreate(&pre_renorm));
	CHECK_CUDA(cudaEventCreate(&post_renorm));

	const size_t shared = LWS * sizeof(float);

	CHECK_CUDA(cudaEventRecord(pre_collapse));

	measure_collapse_16_lmem_k<<< blockNum, LWS, shared >>> (state, d_temp_sums, target_qubit, N);

	CHECK_CUDA(cudaMemcpy(h_temp_sums, d_temp_sums, blockNum * sizeof(float), cudaMemcpyDeviceToHost));

	float p_0 = h_temp_sums[0];
	for (size_t i = 1; i < blockNum; i++)
		p_0 += h_temp_sums[i];

	CHECK_CUDA(cudaEventRecord(post_collapse));

	float ran = (float)rand() / (float)RAND_MAX;
	int measured = (ran < p_0) ? 0 : 1;

	cudaFree(d_temp_sums);
	free(h_temp_sums);

	// renormalize state vector

	float p_measured = (measured == 0) ? p_0 : (1.0 - p_0);

	float norm_factor = 1 / sqrt(p_measured);

	blockNum = (N + LWS - 1) / LWS;

	CHECK_CUDA(cudaEventRecord(pre_renorm));

	measure_renormalize_k << <blockNum, LWS >> > (state, target_qubit, measured, norm_factor, N);

	CHECK_CUDA(cudaEventRecord(post_renorm));

	CHECK_CUDA(cudaEventSynchronize(post_collapse));
	CHECK_CUDA(cudaEventSynchronize(post_renorm));

	CHECK_CUDA(cudaEventElapsedTime(collapse_ms, pre_collapse, post_collapse));
	CHECK_CUDA(cudaEventElapsedTime(renorm_ms, pre_renorm, post_renorm));

	CHECK_CUDA(cudaEventDestroy(pre_collapse));
	CHECK_CUDA(cudaEventDestroy(post_collapse));
	CHECK_CUDA(cudaEventDestroy(pre_renorm));
	CHECK_CUDA(cudaEventDestroy(post_renorm));

	return '0' + measured;

}
