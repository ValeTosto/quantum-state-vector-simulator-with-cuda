#pragma once
#include "common.h"

namespace Gates {
	const complex_t c0 = complex_t(0.0, 0.0);
	const complex_t c1 = complex_t(1.0, 0.0);
	const complex_t ci = complex_t(0.0, 1.0);
	const complex_t cinv_sqrt2 = complex_t(1.0/sqrt(2), 0.0);

	const complex_t X[2][2] = { 
		{c0, c1},
		{c1, c0} 
	};

	const complex_t Y[2][2] = {
		{c0, ci},
		{-ci, c0}
	};

	const complex_t Z[2][2] = {
		{c1, c0},
		{c0, -c1}
	};

	const complex_t H[2][2] = {
		{cinv_sqrt2, cinv_sqrt2},
		{cinv_sqrt2, -cinv_sqrt2}
	};

	const complex_t CNOT[4][4] = {
		{c1, c0, c0, c0},
		{c0, c1, c0, c0},
		{c0, c0, c0, c1},
		{c0, c0, c1, c0}
	};

	const complex_t Toffoli[8][8] = {
		{c1, c0, c0, c0, c0, c0, c0,  c0},
		{c0, c1, c0, c0, c0, c0, c0, c0},
		{c0, c0, c1, c0, c0, c0, c0, c0},
		{c0, c0, c0, c1, c0, c0, c0, c0},
		{c0, c0, c0, c0, c1, c0, c0, c0},
		{c0, c0, c0, c0,c0, c1, c0, c0},
		{c0, c0, c0, c0, c0, c0, c0, c1},
		{c0, c0, c0, c0, c0, c0, c1, c0}
	};

}