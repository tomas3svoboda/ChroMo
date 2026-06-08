#pragma once

#include "definitions.h"

namespace edm_solver {

void q_form(integer n, real *__restrict q, real *__restrict wa);

void q_form_vectorized(integer n, real *__restrict q, real *__restrict wa);

void q_form_omp(integer n, real *__restrict q, real *__restrict wa);

} // namespace edm_solver
