#pragma once

#include "definitions.h"

namespace edm_solver {

void qr_factorization(integer n, real *__restrict a, real *__restrict rdiag,
                      real *__restrict acnorm);

void qr_factorization_vectorized(integer n, real *__restrict a,
                                 real *__restrict rdiag,
                                 real *__restrict acnorm);

void qr_factorization_omp(integer n, real *__restrict a, real *__restrict rdiag,
                          real *__restrict acnorm);

} // namespace edm_solver
