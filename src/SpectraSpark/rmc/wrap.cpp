
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "rmc.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rmc, m) {
  m.doc() = "Reverse Monte Carlo fitting library for in-plane L12 cluster arrangement in LPSO alloy";  // optional module docstring

  py::class_<Simulator>(m, "Simulator")
      .def(py::init<int>())
      .def("set_Lx", &Simulator::set_Lx, "set model size Lx")
      .def("set_Ly", &Simulator::set_Ly, "set model size Ly")
      .def("set_n", &Simulator::set_n, "set number of particles")
      .def("load_exp_data", &Simulator::load_exp_data, "load experimental data from file")
      .def("load_xtl", &Simulator::load_xtl, "load cluster arrangement from xtl file")
      .def("set_q_range", &Simulator::set_q_range, "set q range for evaluation", py::arg("q_min"), py::arg("q_max"))
      .def("init", &Simulator::init, "randomly initialize particle arrangement")
      .def("run", &Simulator::run, "run RMC fitting", py::arg("max_iter"), py::arg("res_thresh"), py::arg("sigma2"), py::arg("move_per_step"))
      .def("save_result", &Simulator::save_result, "save fitting result to file", py::arg("dst"));
}