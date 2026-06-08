#include "edm_solver.h"

#include "powell_solver.h"
#include "residual_function.h"
#include "support_functions.h"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numbers>

void linspace(const real a, const real b, const integer n, real *const array) {
  if (n == 0)
    return;
  if (n == 1) {
    array[0] = b;
    return;
  }
  const real step = (b - a) / (n - 1);
  for (int i = 0; i < n - 1; ++i)
    array[i] = a + step * i;
  array[n - 1] = b;
}

void linspace_no_end(const real a, const real b, const integer n,
                     real *const array) {
  if (n == 0)
    return;
  if (n == 1) {
    array[0] = a;
    return;
  }
  const real step = (b - a) / n;
  for (int i = 0; i < n; ++i)
    array[i] = a + step * i;
}

void linear_approximation(const real *const a, const real a_time,
                          const real *const b, const real b_time, real *const c,
                          const real c_time, const integer N) {
  for (integer n = 0; n < N; n++) {
    const real value_difference = b[n] - a[n];
    const real time_difference = b_time - a_time;
    c[n] = a[n] + value_difference / time_difference * (c_time - a_time);
  }
}

real linear_approximation(const real a, const real a_time, const real b,
                          const real b_time, const real c_time) {
  const real value_difference = b - a;
  const real time_difference = b_time - a_time;
  const real c = a + value_difference / time_difference * (c_time - a_time);
  return c;
}

void normalize_feed_pulse(std::vector<real> &feed,
                          const std::vector<real> &time_vector,
                          const real target_area) {
  // Use trapezoidal rule to compute the time-integral of the discretized feed
  real injected_area = 0;
  const integer n =
      static_cast<integer>(std::min(feed.size(), time_vector.size()));
  if (n < 2)
    return;
  for (integer i = 1; i < n; ++i)
    injected_area += static_cast<real>(0.5) * (time_vector[i] - time_vector[i - 1]) *
                     (feed[i] + feed[i - 1]);

  if (std::abs(injected_area) < 1e-15)
    return;

  const real scale = target_area / injected_area;
  for (real &value : feed)
    value *= scale;
}

real find_geometric_coefficient(const real a1, const real sn, const integer n,
                                real interval_begin, real interval_end) {
  static constexpr integer MAX_STEPS = 1000;
  static constexpr real RELATIVE_TOLERANCE = 1e-8;
  real q = interval_begin + (interval_end - interval_begin) / 2;
  for (integer i = 0; i < MAX_STEPS; i++) {
    if (q == 1)
      return 1;
    const real omqn = (1 - std::pow(q, n));
    const real omq = (1 - q);
    const real rhs = a1 * omqn / omq;
    const real tolerance = std::max(sn, rhs) * RELATIVE_TOLERANCE;
    const real check = std::abs(sn - rhs);
    if (check < tolerance)
      return q;

    if (rhs > sn) {
      interval_end = q;
    } else {
      interval_begin = q;
    }

    q = interval_begin + (interval_end - interval_begin) / 2;
  }
  return q;
}

SolverOutput edm_nonlinear_solver(real flow_rate, real length, real diameter,
                                  real feed_volume, real feed_concentration,
                                  real porosity, real langmuirConst,
                                  real disperCoef, real saturCoef, int Nx,
                                  int Nt, real time, bool debug_print,
                                  bool full) {
  SolverOutput result;
  result.concentration.assign(Nt * Nx, 0.0);
  result.timestamps.assign(Nt, 0.0);

  const real u = (flow_rate * 1000.0 / 3600.0) /
                 ((std::numbers::pi * diameter * diameter / 4.0) * porosity);

  const real feed_length = (feed_volume / flow_rate) * 3600.0;

  constexpr real percentage_of_dense_points = 0.5;
  constexpr real dense_space_ratio = 0.4;

  int Nx_dense = static_cast<int>(std::round(Nx * percentage_of_dense_points));
  int Nx_sparse = Nx - Nx_dense;

  const real dense_length = length * dense_space_ratio;

  std::vector<real> space_vector_x(Nx);
  if (Nx_sparse > 0) {
    linspace_no_end(0.0, dense_length, Nx_dense, space_vector_x.data());
    linspace(dense_length, length, Nx_sparse, space_vector_x.data() + Nx_dense);
  } else
    linspace(0.0, length, Nx, space_vector_x.data());

  std::vector<real> dx(Nx);
  for (int i = 0; i < Nx - 1; ++i)
    dx[i] = space_vector_x[i + 1] - space_vector_x[i];
  dx[Nx - 1] = dx[Nx - 2];

  constexpr real dense_steps_percentage = 0.7;
  const int dense_steps = static_cast<int>(Nt * dense_steps_percentage);
  const real dense_time =
      feed_length + feed_length * ((time / feed_length) / 20.0);

  const integer sparse_steps = Nt - dense_steps;

  std::vector<real> &time_vector = result.timestamps;
  if (sparse_steps > 0) {
    linspace_no_end(0.0, dense_time, dense_steps, time_vector.data());
    linspace(dense_time, time, sparse_steps, time_vector.data() + dense_steps);
  } else
    linspace(0.0, time, Nt, time_vector.data());

  real dt_dense = 0;
  real dt_sparse = 0;
  if (dense_steps >= 2) {
    dt_dense = time_vector[1] - time_vector[0];
  }
  if (sparse_steps >= 2) {
    dt_sparse = time_vector[dense_steps + 1] - time_vector[dense_steps + 0];
  }

  std::vector<real> feed(Nt, 0);
  const auto feed_steps = static_cast<integer>(feed_length / dt_dense);

  feed[0] = feed_concentration / 10.;
  feed[1] = feed_concentration / 5.;
  feed[2] = feed_concentration / 2.;
  feed[3] = feed_concentration / 5. * 4.;
  feed[4] = feed_concentration / 10. * 9.;

  for (integer i = 5; i < feed_steps; ++i)
    feed[i] = feed_concentration;

  feed[feed_steps] = feed_concentration / 10. * 9.;
  feed[feed_steps + 1] = feed_concentration / 5. * 4.;
  feed[feed_steps + 2] = feed_concentration / 2.;
  feed[feed_steps + 3] = feed_concentration / 5.;
  feed[feed_steps + 4] = feed_concentration / 10.;
  normalize_feed_pulse(feed, time_vector, feed_length * feed_concentration);

  const integer lwa = Nx * (3 * Nx + 13);
  std::vector<real> working_array(lwa);
  std::vector<real> output_vector(Nx);

  ResidualContext ctx{nullptr,   dx.data(),  0, porosity, langmuirConst,
                      saturCoef, disperCoef, u, 0};

  std::vector<real> c0(Nx, 0);
  std::vector<real> c1(Nx, 0);
  ctx.c0 = c0.data();
  for (int j = 1; j < Nt; ++j) {
    const real dt_cur = (j < dense_steps) ? dt_dense : dt_sparse;

    ctx.dt = dt_cur;
    ctx.current_feed = feed[j];

    static constexpr real XTOL = 1e-9L;

    edm_solver::powell_caller(residual_func, &ctx, Nx, c1.data(),
                              output_vector.data(), XTOL, working_array.data(),
                              lwa);

    std::memcpy(result.concentration.data() + j * Nx, c1.data(),
                Nx * sizeof(real));
    std::memcpy(c0.data(), c1.data(), Nx * sizeof(real));
  }

  if (debug_print || full) {
    const real feed_mass = feed_volume * feed_concentration;
    real total_mass_output = 0;
    for (integer i = 1; i < Nt; ++i) {
      real actConcOut = result.concentration[i * Nx + Nx - 1];
      real prevConcOut = result.concentration[(i - 1) * Nx + Nx - 1];
      // Use trapezoidal rule: 0.5 * (c[i] + c[i-1]) like Python solver
      real current_mass_output = static_cast<real>(0.5) *
                                 (time_vector[i] - time_vector[i - 1]) *
                                 (actConcOut + prevConcOut) * flow_rate / 3600.0;
      total_mass_output += current_mass_output;
    }
    // Take absolute value of sum, not individual terms
    total_mass_output = std::abs(total_mass_output);
    const real mass_difference = feed_mass - total_mass_output;
    const real relative_mass_difference =
        std::abs(mass_difference * 100 / feed_mass);

    if (debug_print) {
      std::cout << "Feed Mass: " << feed_mass << " mg" << std::endl;
      std::cout << "Outlet Mass: " << total_mass_output << " mg" << std::endl;
      std::cout << "Difference: " << mass_difference << " mg "
                << relative_mass_difference << " %" << std::endl;
    }

    if (full) {
      result.feed = std::move(feed);
      result.relative_mass_difference = relative_mass_difference;
    }
  }

  return result;
}

/**
 * Dynamic solver
 */
SolverOutput edm_nonlinear_dynamic_solver(real flow_rate, real length,
                                          real diameter, real feed_volume,
                                          real feed_concentration,
                                          real porosity, real langmuirConst,
                                          real disperCoef, real saturCoef,
                                          int Nx, int Nt, real experiment_time,
                                          bool debug_print, bool full) {
  SolverOutput result;
  result.concentration.resize(Nt * Nx);
  result.timestamps.resize(Nt);

  real *output_matrix = result.concentration.data();
  real *output_time_vector = result.timestamps.data();

  const real u = (flow_rate * 1000.0 / 3600.0) /
                 ((std::numbers::pi * diameter * diameter / 4.0) * porosity);

  /*
   *  Space dimension preparation
   */
  constexpr real percentage_of_dense_points = 0.5;
  constexpr real dense_space_ratio = 0.4;

  int Nx_dense = static_cast<int>(std::round(Nx * percentage_of_dense_points));
  int Nx_sparse = Nx - Nx_dense;

  const real dense_length = length * dense_space_ratio;

  std::vector<real> space_vector_x(Nx);
  if (Nx_sparse > 0) {
    linspace_no_end(0.0, dense_length, Nx_dense, space_vector_x.data());
    linspace(dense_length, length, Nx_sparse, space_vector_x.data() + Nx_dense);
  } else
    linspace(0.0, length, Nx, space_vector_x.data());

  std::vector<real> dx(Nx);
  for (int i = 0; i < Nx - 1; ++i)
    dx[i] = space_vector_x[i + 1] - space_vector_x[i];
  dx[Nx - 1] = dx[Nx - 2];

  /*
   *  Time dimension preparation
   */
  const real feed_time = (feed_volume / flow_rate) * 3600.0;
  constexpr real FEED_STEPS_PERCENTAGE = 0.5;
  const real dense_time = feed_time;

  const auto output_feed_steps =
      static_cast<integer>(std::round(FEED_STEPS_PERCENTAGE * Nt));
  const auto output_dense_steps =
      static_cast<integer>(std::round(output_feed_steps * 1.1));
  const integer sparse_steps = Nt - output_dense_steps;

  const real dt_dense = dense_time / output_dense_steps;
  const real dt_sparse = (experiment_time - dense_time) / sparse_steps;

  std::vector<real> output_feed(Nt, 0);

  // prepare time vector
  linspace_no_end(0.0, dense_time, output_dense_steps, output_time_vector);
  for (integer i = output_dense_steps; i < Nt - 1; i++) {
    output_time_vector[i] = output_time_vector[i - 1] + dt_sparse;
  }
  output_time_vector[Nt - 1] = experiment_time;

  output_feed[0] = feed_concentration / 10.;
  output_feed[1] = feed_concentration / 5.;
  output_feed[2] = feed_concentration / 2.;
  output_feed[3] = feed_concentration / 5. * 4.;
  output_feed[4] = feed_concentration / 10. * 9.;

  for (integer i = 5; i < output_feed_steps; ++i)
    output_feed[i] = feed_concentration;

  output_feed[output_feed_steps] = feed_concentration / 10. * 9.;
  output_feed[output_feed_steps + 1] = feed_concentration / 5. * 4.;
  output_feed[output_feed_steps + 2] = feed_concentration / 2.;
  output_feed[output_feed_steps + 3] = feed_concentration / 5.;
  output_feed[output_feed_steps + 4] = feed_concentration / 10.;
  normalize_feed_pulse(output_feed, result.timestamps,
                       feed_time * feed_concentration);

  const integer lwa = Nx * (3 * Nx + 13);
  std::vector<real> working_array(lwa);
  std::vector<real> output_vector(Nx);

  ResidualContext context{nullptr,   dx.data(),  0, porosity, langmuirConst,
                          saturCoef, disperCoef, u, 0};

  // maximum allowed time points for dynamic calculation
  const integer max_time_steps = Nt * 2;

  const auto solution_matrix =
      static_cast<real *>(malloc(Nx * max_time_steps * sizeof(real)));
  const auto computation_time_vector =
      static_cast<real *>(malloc(max_time_steps * sizeof(real)));
  const auto c0 = static_cast<real *>(calloc(Nx, sizeof(real)));
  const auto c1 = static_cast<real *>(calloc(Nx, sizeof(real)));

  if (solution_matrix == nullptr || computation_time_vector == nullptr ||
      c0 == nullptr || c1 == nullptr)
    throw std::bad_alloc();

  context.c0 = c0;
  real time_delta = dt_dense;
  computation_time_vector[0] = 0;
  integer current_computation_time_step = 1;

  real avg_difference = 0;

  // how many times was the time
  integer recomputation_count = 0;

  const integer max_dense_steps = output_dense_steps * 2;

  // loop for dense calculation
  constexpr real XTOL = 1e-9;
  for (; current_computation_time_step < max_dense_steps;
       current_computation_time_step++) {
    real current_time =
        computation_time_vector[current_computation_time_step - 1] + time_delta;

    if (current_time >= dense_time)
      break;

    context.dt = time_delta;

    computation_time_vector[current_computation_time_step] = current_time;
    const auto found = std::lower_bound(output_time_vector,
                                        output_time_vector + Nt, current_time);
    integer index = found - output_time_vector;
    assert(index > 0);
    assert(index < Nt);

    context.current_feed = linear_approximation(
        output_feed[index - 1], output_time_vector[index - 1],
        output_feed[index], output_time_vector[index], current_time);

    const int info = edm_solver::powell_caller(residual_func, &context, Nx, c1,
                                               output_vector.data(), XTOL,
                                               working_array.data(), lwa);

    if (time_delta > dt_dense / 2)
      if (current_computation_time_step != max_dense_steps - 1 && info >= 4 &&
          recomputation_count < 4) {
        recomputation_count++;
        time_delta /= 1.1;
        current_computation_time_step--;
        continue;
      }

    recomputation_count = 0;

    std::memcpy(solution_matrix + current_computation_time_step * Nx, c1,
                Nx * sizeof(real));

    std::memcpy(c0, c1, Nx * sizeof(real));
  }

  const integer target_steps =
      current_computation_time_step + Nt - output_feed_steps;

  bool q_calculation_needed = true;
  real q = 1;

  for (; current_computation_time_step < max_time_steps;
       current_computation_time_step++) {
    real current_time =
        computation_time_vector[current_computation_time_step - 1] + time_delta;

    assert(current_time !=
           computation_time_vector[current_computation_time_step - 1]);

    if (current_time >= experiment_time ||
        current_computation_time_step == max_time_steps - 1) {
      current_time = experiment_time;
      context.dt = current_time -
                   computation_time_vector[current_computation_time_step - 1];
    } else
      context.dt = time_delta;

    computation_time_vector[current_computation_time_step] = current_time;
    const auto found = std::lower_bound(output_time_vector,
                                        output_time_vector + Nt, current_time);
    integer index = found - output_time_vector;
    assert(index > 0);
    assert(index < Nt);

    context.current_feed = linear_approximation(
        output_feed[index - 1], output_time_vector[index - 1],
        output_feed[index], output_time_vector[index], current_time);

    const int info = edm_solver::powell_caller(residual_func, &context, Nx, c1,
                                               output_vector.data(), XTOL,
                                               working_array.data(), lwa);

    if (current_computation_time_step != max_time_steps - 1 && info >= 4 &&
        recomputation_count < 4) {
      recomputation_count++;
      time_delta /= 1.1;
      current_computation_time_step--;
      q_calculation_needed = true;
      continue;
    }

    if (q_calculation_needed) {
      q_calculation_needed = false;
      const integer n = target_steps - current_computation_time_step >= 0
                            ? target_steps - current_computation_time_step
                            : 0;
      q = find_geometric_coefficient(time_delta, experiment_time - current_time,
                                     n, 1, experiment_time);
    }

    recomputation_count = 0;

    std::memcpy(solution_matrix + current_computation_time_step * Nx, c1,
                Nx * sizeof(real));

    const real difference = vector_difference_norm(c0, c1, Nx);
    avg_difference =
        (current_computation_time_step * avg_difference + difference) /
        (current_computation_time_step + 1);

    time_delta *= q;

    std::memcpy(c0, c1, Nx * sizeof(real));

    if (current_time == experiment_time) {
      current_computation_time_step++;
      break;
    }
  }

  if (debug_print) {
    real total_mass_output_computation = 0;
    for (integer i = 1; i < current_computation_time_step; ++i) {
      real actConcOut = solution_matrix[i * Nx + Nx - 1];
      real current_mass_output =
          (computation_time_vector[i] - computation_time_vector[i - 1]) *
          flow_rate * actConcOut / 3600.0;
      if (current_mass_output < 0)
        current_mass_output = -current_mass_output;
      total_mass_output_computation += current_mass_output;
    }
    std::cout << "Computation mass output: " << total_mass_output_computation
              << " mg" << std::endl;
  }

  // reuse allocated buffer
  const auto space_vector = c0;

  // Fit the values into the output vector
  for (integer time_point = 0; time_point < Nt; ++time_point) {
    const real c_time = output_time_vector[time_point];
    const real *const end =
        computation_time_vector + current_computation_time_step;
    const real *const found_time_ptr = std::lower_bound(
        static_cast<const real *>(computation_time_vector), end, c_time);
    assert(found_time_ptr != end);

    const auto b_index =
        static_cast<integer>(found_time_ptr - computation_time_vector);

    if (*found_time_ptr == c_time)
      std::memcpy(output_matrix + time_point * Nx,
                  solution_matrix + b_index * Nx, Nx * sizeof(real));
    else {
      const integer a_index = b_index - 1;
      assert(found_time_ptr != computation_time_vector);
      const real a_time = found_time_ptr[-1];
      const real b_time = found_time_ptr[0];
      linear_approximation(solution_matrix + a_index * Nx, a_time,
                           solution_matrix + b_index * Nx, b_time, space_vector,
                           c_time, Nx);
      std::memcpy(output_matrix + time_point * Nx, space_vector,
                  Nx * sizeof(real));
    }
  }

  free(c1);
  free(c0);
  free(computation_time_vector);
  free(solution_matrix);

  if (debug_print || full) {
    const real feed_mass = feed_volume * feed_concentration;
    real total_mass_output = 0;
    for (integer i = 1; i < Nt; ++i) {
      real actConcOut = result.concentration[i * Nx + Nx - 1];
      real prevConcOut = result.concentration[(i - 1) * Nx + Nx - 1];
      // Use trapezoidal rule: 0.5 * (c[i] + c[i-1]) like Python solver
      real current_mass_output = static_cast<real>(0.5) *
                                 (output_time_vector[i] - output_time_vector[i - 1]) *
                                 (actConcOut + prevConcOut) * flow_rate / 3600.0;
      total_mass_output += current_mass_output;
    }
    // Take absolute value of sum, not individual terms
    total_mass_output = std::abs(total_mass_output);
    const real mass_difference = feed_mass - total_mass_output;
    const real relative_mass_difference =
        std::abs(mass_difference * 100 / feed_mass);

    if (debug_print) {
      std::cout << "Feed Mass: " << feed_mass << " mg" << std::endl;
      std::cout << "Outlet Mass: " << total_mass_output << " mg" << std::endl;
      std::cout << "Difference: " << mass_difference << " mg "
                << relative_mass_difference << " %" << std::endl;
    }

    if (full) {
      result.feed = std::move(output_feed);
      result.relative_mass_difference = relative_mass_difference;
    }
  }

  return result;
}
