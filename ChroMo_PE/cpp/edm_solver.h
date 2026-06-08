#pragma once

#include "definitions.h"
#include "residual_function.h"

#include <optional>
#include <vector>

void linspace(real a, real b, int n, real *array);
void linspace_no_end(real a, real b, integer n, real *array);

struct SolverOutput {
  /**
   * Concentration values over time and components.
   * Flattened 2D array of shape (Nt, Nx), stored row-major:
   *   c[i * Nx + j] gives concentration of component j at time step i.
   */
  std::vector<real> concentration;

  /**
   * Time vector (length Nt), corresponding to each row in `c`.
   */
  std::vector<real> timestamps;

  /**
   * Optional feed profile over time (length Nt).
   * Present only if full output was requested (full = true).
   */
  std::optional<std::vector<real>> feed;

  /**
   * Optional relative mass difference.
   * Present only if full output was requested (full = true).
   */
  std::optional<real> relative_mass_difference;
};

void linear_approximation(const real *a, real a_time, const real *b,
                          real b_time, real *c, real c_time, integer N);

real linear_approximation(real a, real a_time, real b,
                          real b_time, real c_time);

void static_inner_solver(integer Nx, integer Nt, ResidualContext ctx,
                         const real *feed, real *output_matrix,
                         integer dense_steps, real dt_dense, real dt_sparse,
                         integer lwa, real *working_array, real *output_vector);

void dynamic_inner_solver(integer Nx, integer Nt, ResidualContext ctx,
                          const real *feed, real *output_matrix,
                          integer dense_steps, real dt_dense, real dt_sparse,
                          integer lwa, real *working_array, real *output_vector,
                          integer max_time_points,
                          const real *output_time_vector, real end_time,
                          integer feed_end_index, real difference_for_increase,
                          real flow_rate);

real find_geometric_coefficient(real a1, real sn, integer n, real interval_begin, real interval_end);

SolverOutput edm_nonlinear_solver(
    real flow_rate = 150,         // Volume flowrate in [mL/h]
    real length = 320,            // Length of the packed section in the column [mm]
    real diameter = 10,           // Column diameter [mm]
    real feed_volume = 4,         // Feed injection volume [mL]
    real feed_concentration = 6,  // Concentration of the balanced component in
                                  // the feed [g/mL] or [mg/mm^3]
    real porosity = 0.4,          // Total porosity of the sorbent packing [-]
    real langmuirConst = 2,       // Langmuir's constant of the linear isotherm [-]
    real disperCoef = 2,          // Axial dispersion coefficient [mm^2/s]
    real saturCoef = 20,          // Saturation Coefficient
    integer Nx = 100,             // Number of spatial differences - Nx
    integer Nt = 30,              // Number of time differences - Nt
    real time = 10800,            // Finite time of the experiment [s]
    bool debug_print = false,     // Activates debug printing
    bool full = false);           // Full output

SolverOutput edm_nonlinear_dynamic_solver(
    real flow_rate = 150,         // Volume flowrate in [mL/h]
    real length = 320,            // Length of the packed section in the column [mm]
    real diameter = 10,           // Column diameter [mm]
    real feed_volume = 4,         // Feed injection volume [mL]
    real feed_concentration = 6,  // Concentration of the balanced component in
                                  // the feed [g/mL] or [mg/mm^3]
    real porosity = 0.4,          // Total porosity of the sorbent packing [-]
    real langmuirConst = 2,       // Langmuir's constant of the linear isotherm [-]
    real disperCoef = 2,          // Axial dispersion coefficient [mm^2/s]
    real saturCoef = 20,          // Saturation Coefficient
    integer Nx = 100,             // Number of spatial differences - Nx
    integer Nt = 30,              // Number of time differences - Nt
    real time = 10800,            // Finite time of the experiment [s]
    bool debug_print = false,     // Activates debug printing
    bool full = false);           // Full output
