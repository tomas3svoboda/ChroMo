/**
 * Functions from CMinpack: https://devernay.github.io/cminpack/
 */

#include "support_functions.h"

namespace edm_solver {

real enorm(const int n, const real *const x) {
  real s1 = 0.;
  real s2 = 0.;
  real s3 = 0.;
  real x1max = 0.;
  real x3max = 0.;
  const real rdwarf =
      std::sqrt(std::numeric_limits<real>::min() * static_cast<real>(1.5L)) *
      static_cast<real>(10.0L);
  const real rgiant =
      std::sqrt(std::numeric_limits<real>::max()) * static_cast<real>(0.1L);
  const real agiant = rgiant / static_cast<real>(n);
  for (int i = 0; i < n; ++i) {
    const real xabs = fabs(x[i]);
    if (xabs >= agiant) {
      /*              sum for large components. */
      if (xabs > x1max) {
        /* Computing 2nd power */
        const real d1 = x1max / xabs;
        s1 = 1 + s1 * (d1 * d1);
        x1max = xabs;
      } else {
        /* Computing 2nd power */
        const real d1 = xabs / x1max;
        s1 += d1 * d1;
      }
    } else if (xabs <= rdwarf) {
      /*              sum for small components. */
      if (xabs > x3max) {
        /* Computing 2nd power */
        const real d1 = x3max / xabs;
        s3 = 1 + s3 * (d1 * d1);
        x3max = xabs;
      } else if (xabs != 0.) {
        /* Computing 2nd power */
        const real d1 = xabs / x3max;
        s3 += d1 * d1;
      }
    } else {
      /*           sum for intermediate components. */
      /* Computing 2nd power */
      s2 += xabs * xabs;
    }
  }

  /*     calculation of norm. */

  real ret_val;
  if (s1 != 0.)
    ret_val = x1max * sqrt(s1 + (s2 / x1max) / x1max);
  else if (s2 != 0.) {
    if (s2 >= x3max)
      ret_val = sqrt(s2 * (1 + (x3max / s2) * (x3max * s3)));
    else
      ret_val = sqrt(x3max * ((s2 / x3max) + (x3max * s3)));
  } else
    ret_val = x3max * sqrt(s3);

  return ret_val;
}

int fdjac1(function_type fcn_nn, void *p, int n, real *x, const real *fvec,
           real *fjac, int ldfjac, int ml, int mu, real epsfcn, real *wa1,
           real *wa2) {
  /* System generated locals */
  int fjac_dim1, fjac_offset;

  /* Local variables */
  real h;
  int i, j, k;
  real eps, temp;
  int msum;
  int iflag = 0;

  /* Parameter adjustments */
  --wa2;
  --wa1;
  --fvec;
  --x;
  fjac_dim1 = ldfjac;
  fjac_offset = 1 + fjac_dim1 * 1;
  fjac -= fjac_offset;

  /* Function Body */

  /*     epsmch is the machine precision. */

  constexpr real epsmch = std::numeric_limits<real>::epsilon();

  eps = sqrt((std::max(epsfcn, epsmch)));
  msum = ml + mu + 1;
  if (msum >= n) {

    /*        computation of dense approximate jacobian. */

    for (j = 1; j <= n; ++j) {
      temp = x[j];
      h = eps * fabs(temp);
      if (h == 0.)
        h = eps;
      x[j] = temp + h;
      /* the last parameter of fcn_nn() is set to 2 to differentiate
         calls made to compute the function from calls made to compute
         the Jacobian (see fcn() in examples/hybdrv.c, and how njev
         is used to compute the number of Jacobian evaluations) */
      iflag = fcn_nn(p, n, &x[1], &wa1[1], 2);
      if (iflag < 0)
        return iflag;
      x[j] = temp;
      for (i = 1; i <= n; ++i)
        fjac[i + j * fjac_dim1] = (wa1[i] - fvec[i]) / h;
    }
    return 0;
  }

  /*        computation of banded approximate jacobian. */

  for (k = 1; k <= msum; ++k) {
    for (j = k; j <= n; j += msum) {
      wa2[j] = x[j];
      h = eps * fabs(wa2[j]);
      if (h == 0.)
        h = eps;
      x[j] = wa2[j] + h;
    }
    iflag = fcn_nn(p, n, &x[1], &wa1[1], 1);
    if (iflag < 0)
      return iflag;
    for (j = k; j <= n; j += msum) {
      x[j] = wa2[j];
      h = eps * fabs(wa2[j]);
      if (h == 0.)
        h = eps;
      for (i = 1; i <= n; ++i) {
        fjac[i + j * fjac_dim1] = 0.;
        if (i >= j - mu && i <= j + ml)
          fjac[i + j * fjac_dim1] = (wa1[i] - fvec[i]) / h;
      }
    }
  }

  return 0;

  /*     last card of subroutine fdjac1. */
}

void dogleg(int n, const real *r, int lr, const real *diag, const real *qtb,
            real delta, real *x, real *wa1, real *wa2) {
  /* System generated locals */
  real d1, d2, d3, d4;

  /* Local variables */
  int i, j, k, l, jj, jp1;
  real sum, temp, alpha, bnorm;
  real gnorm, qnorm;
  real sgnorm;

  /* Parameter adjustments */
  --wa2;
  --wa1;
  --x;
  --qtb;
  --diag;
  --r;
  (void)lr;

  /* Function Body */

  /*     epsmch is the machine precision. */

  constexpr real epsmch = std::numeric_limits<real>::epsilon();

  /*     first, calculate the gauss-newton direction. */

  jj = n * (n + 1) / 2 + 1;
  for (k = 1; k <= n; ++k) {
    j = n - k + 1;
    jp1 = j + 1;
    jj -= k;
    l = jj + 1;
    sum = 0.;
    if (n >= jp1) {
      for (i = jp1; i <= n; ++i) {
        sum += r[l] * x[i];
        ++l;
      }
    }
    temp = r[jj];
    if (temp == 0.) {
      l = j;
      for (i = 1; i <= j; ++i) {
        /* Computing MAX */
        d2 = fabs(r[l]);
        temp = std::max(temp, d2);
        l = l + n - i;
      }
      temp = epsmch * temp;
      if (temp == 0.)
        temp = epsmch;
    }
    x[j] = (qtb[j] - sum) / temp;
  }

  /*     test whether the gauss-newton direction is acceptable. */

  for (j = 1; j <= n; ++j) {
    wa1[j] = 0.;
    wa2[j] = diag[j] * x[j];
  }
  qnorm = enorm(n, &wa2[1]);
  if (qnorm <= delta) {
    return;
  }

  /*     the gauss-newton direction is not acceptable. */
  /*     next, calculate the scaled gradient direction. */

  l = 1;
  for (j = 1; j <= n; ++j) {
    temp = qtb[j];
    for (i = j; i <= n; ++i) {
      wa1[i] += r[l] * temp;
      ++l;
    }
    wa1[j] /= diag[j];
  }

  /*     calculate the norm of the scaled gradient and test for */
  /*     the special case in which the scaled gradient is zero. */

  gnorm = enorm(n, &wa1[1]);
  sgnorm = 0.;
  alpha = delta / qnorm;
  if (gnorm != 0.) {

    /*     calculate the point along the scaled gradient */
    /*     at which the quadratic is minimized. */

    for (j = 1; j <= n; ++j)
      wa1[j] = wa1[j] / gnorm / diag[j];
    l = 1;
    for (j = 1; j <= n; ++j) {
      sum = 0.;
      for (i = j; i <= n; ++i) {
        sum += r[l] * wa1[i];
        ++l;
      }
      wa2[j] = sum;
    }
    temp = enorm(n, &wa2[1]);
    sgnorm = gnorm / temp / temp;

    /*     test whether the scaled gradient direction is acceptable. */

    alpha = 0.;
    if (sgnorm < delta) {

      /*     the scaled gradient direction is not acceptable. */
      /*     finally, calculate the point along the dogleg */
      /*     at which the quadratic is minimized. */

      bnorm = enorm(n, &qtb[1]);
      temp = bnorm / gnorm * (bnorm / qnorm) * (sgnorm / delta);
      /* Computing 2nd power */
      d1 = sgnorm / delta;
      /* Computing 2nd power */
      d2 = temp - delta / qnorm;
      /* Computing 2nd power */
      d3 = delta / qnorm;
      /* Computing 2nd power */
      d4 = sgnorm / delta;
      temp = temp - delta / qnorm * (d1 * d1) +
             sqrt(d2 * d2 + (1 - d3 * d3) * (1 - d4 * d4));
      /* Computing 2nd power */
      d1 = sgnorm / delta;
      alpha = delta / qnorm * (1 - d1 * d1) / temp;
    }
  }

  /*     form appropriate convex combination of the gauss-newton */
  /*     direction and the scaled gradient direction. */

  temp = (1 - alpha) * std::min(sgnorm, delta);
  for (j = 1; j <= n; ++j)
    x[j] = temp * wa1[j] + alpha * x[j];

  /*     last card of subroutine dogleg. */
}

void r1updt(int m, int n, real *s, int ls, const real *u, real *v, real *w,
            int *sing) {
  /* Initialized data */

#define p5 ((real).5)
#define p25 ((real).25)

  /* Local variables */
  int i, j, l, jj, nm1;
  real tan;
  int nmj;
  real cos, sin, tau, temp, cotan;

  /* Parameter adjustments */
  --w;
  --u;
  --v;
  --s;
  (void)ls;

  /* Function Body */

  /*     giant is the largest magnitude. */

  constexpr real giant = std::numeric_limits<real>::max();

  /*     initialize the diagonal element pointer. */

  jj = n * ((m << 1) - n + 1) / 2 - (m - n);

  /*     move the nontrivial part of the last column of s into w. */

  l = jj;
  for (i = n; i <= m; ++i) {
    w[i] = s[l];
    ++l;
  }

  /*     rotate the vector v into a multiple of the n-th unit vector */
  /*     in such a way that a spike is introduced into w. */

  nm1 = n - 1;
  if (nm1 >= 1) {
    for (nmj = 1; nmj <= nm1; ++nmj) {
      j = n - nmj;
      jj -= m - j + 1;
      w[j] = 0.;
      if (v[j] != 0.) {

        /*        determine a givens rotation which eliminates the */
        /*        j-th element of v. */

        if (fabs(v[n]) < fabs(v[j])) {
          cotan = v[n] / v[j];
          sin = p5 / sqrt(p25 + p25 * (cotan * cotan));
          cos = sin * cotan;
          tau = 1.;
          if (fabs(cos) * giant > 1.)
            tau = 1 / cos;
        } else {
          tan = v[j] / v[n];
          cos = p5 / sqrt(p25 + p25 * (tan * tan));
          sin = cos * tan;
          tau = sin;
        }

        /*        apply the transformation to v and store the information */
        /*        necessary to recover the givens rotation. */

        v[n] = sin * v[j] + cos * v[n];
        v[j] = tau;

        /*        apply the transformation to s and extend the spike in w. */

        l = jj;
        for (i = j; i <= m; ++i) {
          temp = cos * s[l] - sin * w[i];
          w[i] = sin * s[l] + cos * w[i];
          s[l] = temp;
          ++l;
        }
      }
    }
  }

  /*     add the spike from the rank 1 update to w. */

  for (i = 1; i <= m; ++i)
    w[i] += v[n] * u[i];

  /*     eliminate the spike. */

  *sing = false;
  if (nm1 >= 1) {
    for (j = 1; j <= nm1; ++j) {
      if (w[j] != 0.) {

        /*        determine a givens rotation which eliminates the */
        /*        j-th element of the spike. */

        if (fabs(s[jj]) < fabs(w[j])) {
          cotan = s[jj] / w[j];
          sin = p5 / sqrt(p25 + p25 * (cotan * cotan));
          cos = sin * cotan;
          tau = 1.;
          if (fabs(cos) * giant > 1.)
            tau = 1 / cos;
        } else {
          tan = w[j] / s[jj];
          cos = p5 / sqrt(p25 + p25 * (tan * tan));
          sin = cos * tan;
          tau = sin;
        }

        /*        apply the transformation to s and reduce the spike in w. */

        l = jj;
        for (i = j; i <= m; ++i) {
          temp = cos * s[l] + sin * w[i];
          w[i] = -sin * s[l] + cos * w[i];
          s[l] = temp;
          ++l;
        }

        /*        store the information necessary to recover the */
        /*        givens rotation. */

        w[j] = tau;
      }

      /*        test for zero diagonal elements in the output s. */

      if (s[jj] == 0.)
        *sing = true;
      jj += m - j + 1;
    }
  }

  /*     move w back into the last column of the output s. */

  l = jj;
  for (i = n; i <= m; ++i) {
    s[l] = w[i];
    ++l;
  }
  if (s[jj] == 0.)
    *sing = true;

  /*     last card of subroutine r1updt. */
}

void r1mpyq(int m, int n, real *a, int lda, const real *v, const real *w) {
  /* System generated locals */
  int a_dim1, a_offset;

  /* Local variables */
  int i, j, nm1, nmj;
  real cos, sin, temp;

  /* Parameter adjustments */
  --w;
  --v;
  a_dim1 = lda;
  a_offset = 1 + a_dim1 * 1;
  a -= a_offset;

  /* Function Body */

  /*     apply the first set of givens rotations to a. */

  nm1 = n - 1;
  if (nm1 < 1)
    return;
  for (nmj = 1; nmj <= nm1; ++nmj) {
    j = n - nmj;
    if (fabs(v[j]) > 1.) {
      cos = 1 / v[j];
      sin = sqrt(1 - cos * cos);
    } else {
      sin = v[j];
      cos = sqrt(1 - sin * sin);
    }
    for (i = 1; i <= m; ++i) {
      temp = cos * a[i + j * a_dim1] - sin * a[i + n * a_dim1];
      a[i + n * a_dim1] = sin * a[i + j * a_dim1] + cos * a[i + n * a_dim1];
      a[i + j * a_dim1] = temp;
    }
  }

  /*     apply the second set of givens rotations to a. */

  for (j = 1; j <= nm1; ++j) {
    if (fabs(w[j]) > 1.) {
      cos = 1 / w[j];
      sin = sqrt(1 - cos * cos);
    } else {
      sin = w[j];
      cos = sqrt(1 - sin * sin);
    }
    for (i = 1; i <= m; ++i) {
      temp = cos * a[i + j * a_dim1] + sin * a[i + n * a_dim1];
      a[i + n * a_dim1] = -sin * a[i + j * a_dim1] + cos * a[i + n * a_dim1];
      a[i + j * a_dim1] = temp;
    }
  }

  /*     last card of subroutine r1mpyq. */
}

} // namespace edm_solver
