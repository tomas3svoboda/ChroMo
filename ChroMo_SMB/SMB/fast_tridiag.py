# fast_tridiag.py
import numpy as np
from numba import njit

@njit
def thomas_precompute(a, b, c):
    N = b.size
    c_ = np.empty_like(c)
    d_ = np.empty_like(b)
    d_[0] = b[0]
    c_[0] = c[0] / d_[0]
    for i in range(1, N-1):
        denom = b[i] - a[i] * c_[i-1]
        d_[i] = denom
        c_[i] = c[i] / denom
    d_[N-1] = b[N-1] - a[N-1] * c_[N-2]
    return c_, d_

@njit
def thomas_solve_from_precomp(a, c_, d_, rhs):
    N = rhs.size
    y = np.empty_like(rhs)
    y[0] = rhs[0] / d_[0]
    for i in range(1, N):
        y[i] = (rhs[i] - a[i] * y[i-1]) / d_[i]
    x = np.empty_like(rhs)
    x[N-1] = y[N-1]
    for i in range(N-2, -1, -1):
        x[i] = y[i] - c_[i] * x[i+1]
    return x
