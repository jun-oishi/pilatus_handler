#include <cmath>
#include <Eigen/Dense>
#include <tuple>

using namespace Eigen;
using namespace std;

tuple<VectorXd, VectorXd> integrate(MatrixXd const &img, tuple<double, double> center) {
  int width = img.cols(), height = img.rows();
  double cx = get<0>(center), cy = get<1>(center);

  MatrixXd q(width, height);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      q(i, j) = sqrt(pow(i - cx, 2) + pow(j - cy, 2));
    }
  }

  int len = ceil(q.maxCoeff());
  VectorXi vq = VectorXi::LinSpaced(len, 0, len - 1);
  VectorXd vi = VectorXd::Zero(len);
  VectorXi count = VectorXi::Zero(len);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (img(i, j) > 0) {
        int idx = floor(q(j, i));
        vi(idx) += img(i, j);
        count(idx)++;
      }
    }
  }

  for (int i = 0; i < len; i++) {
    if (count(i) > 0) {
      vi(i) /= count(i);
    }
  }

  VectorXd vqd = VectorXd::LinSpaced(len+0.5, 0.5, len - 1);
  return make_tuple(vq, vi);
}