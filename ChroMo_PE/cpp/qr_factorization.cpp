/**
 *  Based on CMinpack: https://devernay.github.io/cminpack/
 */

#include "qr_factorization.h"
#include "support_functions.h"

#include <cstring>

namespace edm_solver {

void qr_factorization(const integer n, real *__restrict const a,
                      real *__restrict const rdiag,
                      real *__restrict const acnorm) {
  if (n > 300)
    qr_factorization_omp(n, a, rdiag, acnorm);
  else
    qr_factorization_vectorized(n, a, rdiag, acnorm);
}

void qr_factorization_vectorized(const integer n, real *__restrict const a,
                                 real *__restrict const rdiag,
                                 real *__restrict const acnorm) {
  /* compute the initial column norms and initialize several arrays. */
  for (integer j = 0; j < n; ++j) {
    const auto norm = enorm(n, a + j * n);
    acnorm[j] = norm;
  }
  std::memcpy(rdiag, acnorm, n * sizeof(real));

  /* reduce a to r with householder transformations. */
  for (integer j = 0; j < n; ++j) {
    /* compute the householder transformation to reduce the */
    /* j-th column of a to a multiple of the j-th unit vector. */
    real ajnorm = enorm(n - (j + 1) + 1, &a[j + j * n]);
    if (ajnorm != 0.) {
      if (a[j + j * n] < 0.)
        ajnorm = -ajnorm;
      for (int i = j; i < n; ++i)
        a[i + j * n] /= ajnorm;
      a[j + j * n] += 1;

      /* apply the transformation to the remaining columns */
      /* and update the norms. */
      if (n > j + 1) {
        for (int k = j + 1; k < n; ++k) {
          real sum = 0.;
          for (int i = j; i < n; ++i)
            sum += a[i + j * n] * a[i + k * n];
          const real temp = sum / a[j + j * n];
          for (int i = j; i < n; ++i)
            a[i + k * n] -= temp * a[i + j * n];
        }
      }
    }
    rdiag[j] = -ajnorm;
  }
}

void qr_factorization_omp(const integer n, real *__restrict const a,
                          real *__restrict const rdiag,
                          real *__restrict const acnorm) {
  /* compute the initial column norms and initialize arrays. */
  for (int j = 0; j < n; ++j) {
    const auto norm = enorm(n, a + j * n);
    acnorm[j] = norm;
  }
  std::memcpy(rdiag, acnorm, n * sizeof(real));

  /* reduce a to r with Householder transformations. */
  for (int j = 0; j < n; ++j) {

    /* compute the Householder transformation for column j. */
    real ajnorm = enorm(n - j, &a[j + j * n]); // m-(j+1)+1 == m-j
    if (ajnorm != 0.) {
      if (a[j + j * n] < 0.)
        ajnorm = -ajnorm;

      /* normalize column j (serial) */
      for (int i = j; i < n; ++i)
        a[i + j * n] /= ajnorm;
      a[j + j * n] += 1;

      /* apply the transformation to remaining columns in parallel */
      if (n > j + 1) {
#pragma omp parallel for schedule(guided, 8) default(none) shared(j, n, a)
        for (int k = j + 1; k < n; ++k) {
          /* compute dot = a[j:m-1, j]^T * a[j:m-1, k] */
          real sum = 0.;
          for (int i = j; i < n; ++i)
            sum += a[i + j * n] * a[i + k * n];

          const real temp = sum / a[j + j * n];

          /* update column k */
          for (int i = j; i < n; ++i)
            a[i + k * n] -= temp * a[i + j * n];
        } // end parallel for k
      }
    }
    rdiag[j] = -ajnorm;
  }
}

} // namespace edm_solver
