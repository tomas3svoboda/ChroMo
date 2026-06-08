#include "definitions.h"

real vector_difference_norm(const real *const a, const real *const b, const integer n) {
  real result = 0;
  for (integer i = 0; i < n; i++) {
    result += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return std::sqrt(result);
}
