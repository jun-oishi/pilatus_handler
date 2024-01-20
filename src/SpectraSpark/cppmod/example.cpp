

#include "example.hpp"

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

void savetxt(string filename, Eigen::MatrixXf m, string header) {
  ofstream file(filename);
  if (file.is_open()) {
    file << header << endl;
    file << m << endl;
    file.close();
  } else {
    throw std::runtime_error("Unable to open file " + filename);
  }
}

/**
 * @brief Load a matrix from a text file
 * @param filename The name of the file
 * @param skiprows The number of header lines to skip
 * @return The loaded matrix
 */
Eigen::MatrixXf loadtxt(string filename, int skiprows) {
  ifstream file(filename);
  if (file.is_open()) {
    string line;
    for (int i = 0; i < skiprows; i++) getline(file, line);
    vector<vector<float>> data;
    while (getline(file, line)) {
      vector<float> row;
      stringstream iss(line);
      float val;
      while (iss >> val) row.push_back(val);
      data.push_back(row);
    }
    file.close();
    Eigen::MatrixXf m(data.size(), data[0].size());
    for (uint i = 0; i < data.size(); i++)
      for (uint j = 0; j < data[0].size(); j++) m(i, j) = data[i][j];
    return m;
  } else {
    throw std::runtime_error("Unable to open file " + filename);
  }
}

Eigen::MatrixXi shrink(Eigen::MatrixXi m, float factor) {
  if (factor <= 0) throw std::runtime_error("factor must be positive");
  if (factor >= 1) return m;
  int rows = m.rows() * factor;
  int cols = m.cols() * factor;
  Eigen::MatrixXi ret(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++)
      ret(i, j) =
          m.block(i / factor, j / factor, 1 / factor, 1 / factor).mean();
  }
  return ret;
}

double sum(Eigen::VectorXd v) {
  double sum = 0;
  double c = 0;
  for (int i = 0; i < v.size(); i++) {
    double y = v(i) - c;
    double t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}

ExampleClass::ExampleClass(int i, int n) {
  val = i;
  vec = Eigen::VectorXd::Zero(n);
};

void ExampleClass::increment() { val++; }

int ExampleClass::get_val() { return val; }

Eigen::VectorXd ExampleClass::get_vec() { return vec; }

void ExampleClass::set_vec(Eigen::VectorXd *v) { vec = *v; }

void ExampleClass::set_el(int i, double val) { vec(i) = val; }

}  // namespace example