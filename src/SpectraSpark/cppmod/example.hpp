

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;

namespace example {
/**
 * @brief Save a matrix to a text file
 * @param filename The name of the file
 * @param m The matrix to save
 * @param header The header of the file
 */
#include <fstream>  // Include the <fstream> header file

void savetxt(string filename, Eigen::MatrixXf m, string header);

/**
 * @brief Load a matrix from a text file
 * @param filename The name of the file
 * @param skiprows The number of header lines to skip
 * @return The loaded matrix
 */
Eigen::MatrixXf loadtxt(string filename, int skiprows);

Eigen::MatrixXi shrink(Eigen::MatrixXi m, float factor);

double sum(Eigen::VectorXd v);

class ExampleClass {
 private:
  int val;
  Eigen::VectorXd vec;

 public:
  ExampleClass(int i, int n);
  void increment();
  int get_val();
  Eigen::VectorXd get_vec();
  void set_vec(Eigen::VectorXd *v);
  void set_el(int i, double val);
};

}  // namespace example