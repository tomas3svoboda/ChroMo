#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "edm_solver.h"

namespace py = pybind11;

PYBIND11_MODULE(edm_nonlinear_solver, m, py::mod_gil_not_used()) {
  m.doc() = "EDM Solver with Nonlinear Langmuir's isotherm";

  py::class_<SolverOutput>(m, "SolverResult")
      .def_readonly("concentration", &SolverOutput::concentration)
      .def_readonly("timestamps", &SolverOutput::timestamps)
      .def_readonly("feed", &SolverOutput::feed)
      .def_readonly("relative_mass_difference",
                    &SolverOutput::relative_mass_difference);

  m.def("edm_nonlinear_solver", &edm_nonlinear_solver,
        "EDM Solver with Nonlinear Langmuir's isotherm");

  m.def("edm_dynamic_solver", &edm_nonlinear_dynamic_solver,
        "EDM Solver with Nonlinear Langmuir's isotherm, dynamic");
}
