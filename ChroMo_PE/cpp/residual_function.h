#pragma once

#include "definitions.h"

struct ResidualContext {
  const real *c0;
  const real *dx;
  real current_feed;
  real porosity;
  real K;
  real Q;
  real D;
  real u;
  real dt;
};

int residual_func(void *p, int n, const real *x, real *fvec, int);
