#include "qform.h"

#include <algorithm>
#include <cstring>

namespace edm_solver {

void q_form(const integer n, real *q, real *wa) {
  if (n > 300)
    q_form_omp(n, q, wa);
  else
    q_form_vectorized(n, q, wa);
}

void q_form_vectorized(const integer n, real *__restrict const q,
                       real *__restrict const wa) {
  /* zero out upper triangle of q in the first min(m,n) columns. */
  if (n >= 1) {
    for (integer j = 1; j < n; ++j) {
      std::memset(q + j * n, 0, j * sizeof(real));
    }
  }

  /* accumulate q from its factored form. */
  for (integer l = 0; l < n; ++l) {
    const integer k = n - l - 1;
    std::memcpy(wa + k, q + k + k * n, (n - k) * sizeof(real));
    std::memset(q + k + k * n, 0, (n - k) * sizeof(real));
    q[k + k * n] = 1.;
    if (wa[k] != 0.) {
      for (integer j = k; j < n; ++j) {
        real sum = 0.;
        for (integer i = k; i < n; ++i)
          sum += q[i + j * n] * wa[i];
        const real temp = sum / wa[k];
        for (integer i = k; i < n; ++i)
          q[i + j * n] -= temp * wa[i];
      }
    }
  }
}

void q_form_omp(const integer n, real *__restrict const q,
                real *__restrict const wa) {
  /* zero out upper triangle of q in the first min(m,n) columns. */
  if (n >= 2) {
    for (integer j = 1; j < n; ++j) {
      std::memset(q + j * n, 0, j * sizeof(real));
    }
  }

  /* accumulate q from its factored form. */
  for (integer l = 0; l < n; ++l) {
    const integer k = n - l - 1;
    std::memcpy(wa + k, q + k + k * n, (n - k) * sizeof(real));
    std::memset(q + k + k * n, 0, (n - k) * sizeof(real));
    q[k + k * n] = 1.;
    if (wa[k] != 0.) {
#pragma omp parallel for schedule(guided, 8) default(none) shared(k, n, q, wa)
      for (integer j = k; j < n; ++j) {
        real sum = 0.;
        for (integer i = k; i < n; ++i)
          sum += q[i + j * n] * wa[i];
        const real temp = sum / wa[k];
        for (integer i = k; i < n; ++i)
          q[i + j * n] -= temp * wa[i];
      }
    }
  }
}

} // namespace edm_solver
