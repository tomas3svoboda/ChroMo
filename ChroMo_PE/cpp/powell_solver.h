#pragma once

#include "definitions.h"

namespace edm_solver {

int powell_caller(function_type fcn_nn, void *p, int n, real *x, real *fvec,
                  real tol, real *wa, int lwa);

int powell_solver_optimized(function_type residual_function,
                            void *data_for_residual_function, int n, real *x,
                            real *fvec, real xtol, int maxfev, int ml, int mu,
                            real epsfcn, real *diag, real factor, int nprint,
                            int *nfev, real *fjac, real *r, int lr, real *qtf,
                            real *working_array_1, real *working_array_2,
                            real *working_array_3, real *working_array_4);

} // namespace edm_solver
