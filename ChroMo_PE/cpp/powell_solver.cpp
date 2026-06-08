/**
 *  Based on CMinpack: https://devernay.github.io/cminpack/
 */

#include "powell_solver.h"

#include "qform.h"
#include "qr_factorization.h"
#include "support_functions.h"

#include <algorithm>
#include <iostream>

namespace edm_solver {

int powell_caller(function_type fcn_nn, void *p, int n, real *x, real *fvec,
                  real tol, real *wa, int lwa) {
  if (n <= 0 || tol < 0. || lwa < n * (n * 3 + 13) / 2)
    return 0;

  const real factor = 100.;
  int maxfev = (n + 1) * 200;
  real xtol = tol;
  int ml = n - 1;
  int mu = n - 1;
  real epsfcn = 0.;
  for (int j = 0; j < n; ++j)
    wa[j] = 1.;
  int nprint = 0;
  int lr = n * (n + 1) / 2;
  int index = n * 6 + lr;
  int nfev;

  int info = powell_solver_optimized(
      fcn_nn, p, n, x, fvec, xtol, maxfev, ml, mu, epsfcn, wa, factor, nprint,
      &nfev, &wa[index + 1], &wa[n * 6 + 1], lr, &wa[n], &wa[(n << 1)],
      &wa[n * 3], &wa[(n << 2)], &wa[n * 5]);
  if (info == 5)
    info = 4;

  return info;
}

#undef p5

int powell_solver_optimized(function_type residual_function,
                            void *data_for_residual_function, int n, real *x,
                            real *fvec, real xtol, int maxfev, int ml, int mu,
                            real epsfcn, real *diag, real factor, int nprint,
                            int *nfev, real *fjac, real *r, int lr, real *qtf,
                            real *working_array_1, real *working_array_2,
                            real *working_array_3, real *working_array_4) {
  const int ldfjac = n;

  constexpr real p1 = 0.1L;
  constexpr real p5 = .5L;
  constexpr real p001 = .001L;

  int fjac_dim1, fjac_offset, i1;
  real d1;

  int i, j, l, jm1;
  int iter;
  real temp;
  int msum, iflag;
  real delta = 0.;
  bool jeval;
  int ncsuc;
  real ratio;
  real fnorm;
  real pnorm, xnorm = 0., fnorm1;
  int nslow1, nslow2;
  int ncfail;
  real actred, epsmch, prered;

  --working_array_4;
  --working_array_3;
  --working_array_2;
  --working_array_1;
  --qtf;
  --diag;
  --fvec;
  --x;
  fjac_dim1 = ldfjac;
  fjac_offset = 1 + fjac_dim1 * 1;
  fjac -= fjac_offset;
  --r;

  epsmch = std::numeric_limits<real>::epsilon();

  int info = 0;
  iflag = 0;
  *nfev = 0;

  /*     check the input parameters for errors. */

  if (n <= 0 || xtol < 0. || maxfev <= 0 || ml < 0 || mu < 0 || factor <= 0. ||
      ldfjac < n || lr < n * (n + 1) / 2) {
    throw std::runtime_error("Invalid arguments");
  }

  for (j = 1; j <= n; ++j)
    if (diag[j] <= 0.)
      throw std::runtime_error("Digonal");

  /*     evaluate the function at the starting point */
  /*     and calculate its norm. */

  iflag = residual_function(data_for_residual_function, n, &x[1], &fvec[1], 1);
  *nfev = 1;
  if (iflag < 0)
    goto TERMINATE;
  fnorm = enorm(n, &fvec[1]);

  /*     determine the number of calls to fcn needed to compute */
  /*     the jacobian matrix. */

  /* Computing MIN */
  i1 = ml + mu + 1;
  msum = std::min(i1, n);

  /*     initialize iteration counter and monitors. */

  iter = 1;
  ncsuc = 0;
  ncfail = 0;
  nslow1 = 0;
  nslow2 = 0;

  /*     beginning of the outer loop. */
  for (;;) {
    jeval = true;

    /*        calculate the jacobian matrix. */
    iflag = fdjac1(residual_function, data_for_residual_function, n, &x[1],
                   &fvec[1], &fjac[fjac_offset], ldfjac, ml, mu, epsfcn,
                   &working_array_1[1], &working_array_2[1]);
    *nfev += msum;
    if (iflag < 0)
      goto TERMINATE;

    /*        compute the qr factorization of the jacobian. */

    if (iter == 1) {
      /*        on the first iteration, calculate the norm of the scaled x */
      /*        and initialize the step bound delta. */
      for (j = 1; j <= n; ++j) {
        working_array_3[j] = diag[j] * x[j];
      }
      xnorm = enorm(n, &working_array_3[1]);
      delta = factor * xnorm;
      if (delta == 0.)
        delta = factor;
    }

    int sing = false;

    qr_factorization(n, &fjac[fjac_offset], &working_array_1[1],
                     &working_array_2[1]);
    for (i = 1; i <= n; ++i) {
      qtf[i] = fvec[i];
    }
    for (j = 1; j <= n; ++j) {
      if (fjac[j + j * fjac_dim1] != 0.) {
        real sum = 0.;
        for (i = j; i <= n; ++i) {
          sum += fjac[i + j * fjac_dim1] * qtf[i];
        }
        temp = -sum / fjac[j + j * fjac_dim1];
        for (i = j; i <= n; ++i) {
          qtf[i] += fjac[i + j * fjac_dim1] * temp;
        }
      }
    }

    /*        copy the triangular factor of the qr factorization into r. */
    for (j = 1; j <= n; ++j) {
      l = j;
      jm1 = j - 1;
      if (jm1 >= 1) {
        for (i = 1; i <= jm1; ++i) {
          r[l] = fjac[i + j * fjac_dim1];
          l = l + n - i;
        }
      }
      r[l] = working_array_1[j];
      if (working_array_1[j] == 0.)
        sing = true;
    }

    /*        accumulate the orthogonal factor in fjac. */
    q_form(n, &fjac[fjac_offset], &working_array_1[1]);

    /*        beginning of the inner loop. */

    for (;;) {
      /*           if requested, call fcn to enable printing of iterates. */

      if (nprint > 0) {
        iflag = 0;
        if ((iter - 1) % nprint == 0)
          iflag = residual_function(data_for_residual_function, n, &x[1],
                                    &fvec[1], 0);
        if (iflag < 0)
          goto TERMINATE;
      }

      /*           determine the direction p. */

      dogleg(n, &r[1], lr, &diag[1], &qtf[1], delta, &working_array_1[1],
             &working_array_2[1], &working_array_3[1]);

      /*           store the direction p and x + p. calculate the norm of p. */

      for (j = 1; j <= n; ++j) {
        working_array_1[j] = -working_array_1[j];
        working_array_2[j] = x[j] + working_array_1[j];
        working_array_3[j] = diag[j] * working_array_1[j];
      }
      pnorm = enorm(n, &working_array_3[1]);

      /*           on the first iteration, adjust the initial step bound. */

      if (iter == 1)
        delta = std::min(delta, pnorm);

      /*           evaluate the function at x + p and calculate its norm. */

      iflag = residual_function(data_for_residual_function, n,
                                &working_array_2[1], &working_array_4[1], 1);
      ++(*nfev);
      if (iflag < 0)
        goto TERMINATE;
      fnorm1 = enorm(n, &working_array_4[1]);

      /*           compute the scaled actual reduction. */

      actred = -1.;
      if (fnorm1 < fnorm) {
        /* Computing 2nd power */
        d1 = fnorm1 / fnorm;
        actred = 1 - d1 * d1;
      }

      /*           compute the scaled predicted reduction. */

      l = 1;
      for (i = 1; i <= n; ++i) {
        real sum = 0.;
        for (j = i; j <= n; ++j) {
          sum += r[l] * working_array_1[j];
          ++l;
        }
        working_array_3[i] = qtf[i] + sum;
      }
      temp = enorm(n, &working_array_3[1]);
      prered = 0.;
      if (temp < fnorm) {
        /* Computing 2nd power */
        d1 = temp / fnorm;
        prered = 1 - d1 * d1;
      }

      /*           compute the ratio of the actual to the predicted */
      /*           reduction. */

      ratio = 0.;
      if (prered > 0.)
        ratio = actred / prered;

      /*           update the step bound. */

      if (ratio < p1) {
        ncsuc = 0;
        ++ncfail;
        delta = p5 * delta;
      } else {
        ncfail = 0;
        ++ncsuc;
        if (ratio >= p5 || ncsuc > 1) {
          /* Computing MAX */
          d1 = pnorm / p5;
          delta = std::max(delta, d1);
        }
        if (fabs(ratio - 1) <= p1)
          delta = pnorm / p5;
      }

      /*           test for successful iteration. */
      constexpr real p0001 = 1e-4L;
      if (ratio >= p0001) {

        /*           successful iteration. update x, fvec, and their norms. */

        for (j = 1; j <= n; ++j) {
          x[j] = working_array_2[j];
          working_array_2[j] = diag[j] * x[j];
          fvec[j] = working_array_4[j];
        }
        xnorm = enorm(n, &working_array_2[1]);
        fnorm = fnorm1;
        ++iter;
      }

      /*           determine the progress of the iteration. */

      ++nslow1;
      if (actred >= p001)
        nslow1 = 0;
      if (jeval)
        ++nslow2;
      if (actred >= p1)
        nslow2 = 0;

      /*           test for convergence. */

      if (delta <= xtol * xnorm || fnorm == 0.)
        info = 1;
      if (info != 0)
        goto TERMINATE;

      /*           tests for termination and stringent tolerances. */

      if (*nfev >= maxfev)
        info = 2;
      /* Computing MAX */
      d1 = p1 * delta;
      if (p1 * std::max(d1, pnorm) <= epsmch * xnorm)
        info = 3;
      if (nslow2 == 5)
        info = 4;
      if (nslow1 == 10)
        info = 5;
      if (info != 0)
        goto TERMINATE;

      /*           criterion for recalculating jacobian approximation */
      /*           by forward differences. */

      if (ncfail == 2)
        break;

      /*           calculate the rank one modification to the jacobian */
      /*           and update qtf if necessary. */

      for (j = 1; j <= n; ++j) {
        real sum = 0.;
        for (i = 1; i <= n; ++i) {
          sum += fjac[i + j * fjac_dim1] * working_array_4[i];
        }
        working_array_2[j] = (sum - working_array_3[j]) / pnorm;
        working_array_1[j] = diag[j] * (diag[j] * working_array_1[j] / pnorm);
        if (ratio >= p0001)
          qtf[j] = sum;
      }

      /*           compute the qr factorization of the updated jacobian. */

      r1updt(n, n, &r[1], lr, &working_array_1[1], &working_array_2[1],
             &working_array_3[1], &sing);
      r1mpyq(n, n, &fjac[fjac_offset], ldfjac, &working_array_2[1],
             &working_array_3[1]);
      r1mpyq(1, n, &qtf[1], 1, &working_array_2[1], &working_array_3[1]);

      /*           end of the inner loop. */

      jeval = false;
    }
    /*        end of the outer loop. */
  }
TERMINATE:

  /*     termination, either normal or user imposed. */

  if (iflag < 0)
    info = iflag;
  if (nprint > 0)
    residual_function(data_for_residual_function, n, &x[1], &fvec[1], 0);

  return info;

  /*     last card of subroutine hybrd. */
}

} // namespace edm_solver
