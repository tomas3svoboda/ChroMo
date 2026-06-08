#include "residual_function.h"

int residual_func(void *p, int n, const real *__restrict x,
                  real *__restrict fvec, const int iflag) {
  const ResidualContext *__restrict const ctx =
      static_cast<ResidualContext *>(p);
  const real *__restrict const c0 = ctx->c0;
  const real *__restrict const dx = ctx->dx;

  constexpr auto eps = static_cast<real>(1e-10L);
  constexpr auto ONE = static_cast<real>(1.0L);
  constexpr auto TWO = static_cast<real>(2.0L);
  constexpr auto HALF = static_cast<real>(0.5L);

  fvec[0] = (((c0[1] - c0[0]) / dx[0]) + ((x[1] - x[0]) / dx[0])) * HALF -
            (ctx->u / ctx->D) * (x[0] - ctx->current_feed);

  for (int i = 1; i < n - 1; i++) {
    const real denominator_0 =
        ((ONE - ctx->porosity) * ctx->Q * ctx->K) /
        ((((ctx->K * c0[i] + ONE) * (ctx->K * c0[i] + ONE)) * ctx->porosity) +
         eps);

    const real denominator_1 =
        ((ONE - ctx->porosity) * ctx->Q * ctx->K) /
        ((((ctx->K * x[i] + ONE) * (ctx->K * x[i] + ONE)) * ctx->porosity) +
         eps);

    const real d2c0 = (c0[i - 1] - TWO * c0[i] + c0[i + 1]) / (dx[i] * dx[i]);
    const real d2c1 = (x[i - 1] - TWO * x[i] + x[i + 1]) / (dx[i] * dx[i]);

    const real dc0 = (c0[i + 1] - c0[i - 1]) / (TWO * dx[i]);
    const real dc1 = (x[i + 1] - x[i - 1]) / (TWO * dx[i]);

    const real time_derivation = (x[i] - c0[i]) / ctx->dt;

    const real disper =
        ((ctx->D / denominator_0 * d2c0) + (ctx->D / denominator_1 * d2c1)) *
        HALF;

    const real conv =
        ((ctx->u / denominator_0 * dc0) + (ctx->u / denominator_1 * dc1)) *
        HALF;

    fvec[i] = disper - conv - time_derivation;
  }

  if (const int last = n - 1; last > 0)
    fvec[last] = (((c0[last] - c0[last - 1]) / dx[last]) +
                  ((x[last] - x[last - 1]) / dx[last])) *
                 HALF;

  return iflag;
}
