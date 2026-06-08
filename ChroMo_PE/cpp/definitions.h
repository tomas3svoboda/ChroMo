#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

using real = double;
using integer = int;

real vector_difference_norm(const real *a, const real *b, integer n);

using function_type = int (*)(void *data_for_residual_function, int n,
                              const real *x, real *fvec, int iflag);
