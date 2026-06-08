#pragma once

#include "definitions.h"

namespace edm_solver {

real enorm(int n, const real *x);

int fdjac1(function_type fcn_nn, void *p, int n, real *x, const real *fvec,
           real *fjac, int ldfjac, int ml, int mu, real epsfcn, real *wa1,
           real *wa2);

void dogleg(int n, const real *r, int lr, const real *diag, const real *qtb,
            real delta, real *x, real *wa1, real *wa2);

void r1updt(int m, int n, real *s, int ls, const real *u, real *v, real *w,
            int *sing);

void r1mpyq(int m, int n, real *a, int lda, const real *v, const real *w);

} // namespace edm_solver
