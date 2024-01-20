
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "example.hpp"

namespace py = pybind11;

PYBIND11_MODULE(example, m) {
  m.doc() = "pybind11 example plugin 1";  // optional module docstring

  m.def("savetxt", &example::savetxt,
        "A function to save a matrix to a text file");
  m.def("loadtxt", &example::loadtxt,
        "A function to load a matrix from a text file");
  m.def("shrink", &example::shrink, "A function to shrink a matrix");
  m.def("sum", &example::sum, "A function to sum a matrix");

  py::class_<example::ExampleClass>(m, "ExampleClass")
      .def(py::init<int, int>())
      .def("increment", &example::ExampleClass::increment)
      .def("get_val", &example::ExampleClass::get_val)
      .def("get_vec", &example::ExampleClass::get_vec)
      .def("set_vec", &example::ExampleClass::set_vec)
      .def("set_el", &example::ExampleClass::set_el);
}