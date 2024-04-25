#include "rmc2d.hpp"
#include "rmc2d_util.hpp"
#include <fstream>
#include <iostream>

namespace RMC {

using namespace std;

#ifndef SQ
#define SQ(x) ((x) * (x))
#endif
#define LAUE(x) (SQ(sin(5 * (x))) / SQ(sin(x)))

#ifndef DELETE
#define DELETE(x) delete[] x; x = nullptr;
#endif

#define PRINT_VEC(st, n, vec) \
  st << "  "; \
  for (int __i_##vec = 0; __i_##vec < n; __i_##vec++) { \
    st << vec[__i_##vec] << "  "; \
  } \
  st << endl;

void save_config(const string &filename, const RMC::Simulator2d &sim) {
  int n, La, Lb;
  int *a=nullptr, *b=nullptr;
  sim.get_config(n, La, Lb, a, b);

  ofstream fp(filename);
  fp << "comment:" << endl << "" << endl;
  fp << "La  Lb  N:" << endl
     << "  " << La << " " << Lb << " " << n << endl;
  fp << "a:" << endl;
  PRINT_VEC(fp, n, a);
  fp << "b:" << endl;
  PRINT_VEC(fp, n, b);
  fp.close();

  cout << filename << " saved" << endl;

  DELETE(a);
  DELETE(b);
}

void load_config(const string &filename,
                 int *n, int *La, int *Lb, int *a, int *b) {
  ifstream fp(filename);
  string line;
  getline(fp, line);  // comment:
  getline(fp, line);  // "comment content"
  getline(fp, line);  // La Lb N:
  fp >> *La >> *Lb >> *n;
  a = new int[*n];
  b = new int[*n];
  getline(fp, line);  // a:
  for (int i = 0; i < *n; i++) fp >> a[i];
  getline(fp, line);  // b:
  for (int i = 0; i < *n; i++) fp >> b[i];
  fp.close();
  return;
}

void save_i(const string &filename, int w, int h,
            double *qx, double *qy, double *intensity) {
  ofstream fp(filename);
  fp << "qx:" << endl << "  " << w << endl;
  PRINT_VEC(fp, w, qx);
  fp << "qy:" << endl << "  " << h << endl;
  PRINT_VEC(fp, h, qy);
  fp << "i:" << endl;
  for (int i = 0; i < h; i++) {
    double *head = intensity + i * w;
    PRINT_VEC(fp, w, head);
  }
  fp.close();
  cout << filename << " saved" << endl;
  return;
}

void load_i(const string &filename, int &width, int &height, double *&qx,
            double *&qy, double *&intensity) {
  ifstream fp(filename);
  string _;
  fp >> _; // qx:
  fp >> width;
  qx = new double[width];
  for (int i = 0; i < width; i++) fp >> qx[i];
  fp >> _; // qy:
  fp >> height;
  qy = new double[height];
  for (int i = 0; i < height; i++) fp >> qy[i];
  fp >> _; // i:
  intensity = new double[width * height];
  for (int i = 0; i < height; i++) {
    double *head = intensity + i * width;
    for (int j = 0; j < width; j++) fp >> head[j];
  }
  fp.close();
  return;
}

void save_result(const string &filename, const Simulator2d &sim,
                 int n_step, double *const res_hist) {
  string dst;
  dst = filename + ".dat";
  int w, h;
  double *qx=nullptr, *qy=nullptr, *i_fit=nullptr;
  sim.get_i_sim(w, h, qx, qy, i_fit);
  save_i(dst, w, h, qx, qy, i_fit);
  DELETE(qx);
  DELETE(qy);
  DELETE(i_fit);

  dst = filename + ".conf";
  save_config(dst, sim);

  dst = filename + ".log";
  ofstream fp(dst);
  fp << "step  residual" << endl;
  for (int i = 0; i < n_step + 1; i++) {
    fp << i << "  " << res_hist[i] << endl;
  }
  fp.close();
  cout << dst << " saved" << endl;
}

void gen_sample_data(int w, int h, double *qx, double *qy, double *i_exp,
                  double Lx, double Ly) {
  for (int i = 0; i < h; i++) {
    double _qy = qy[i];
    for (int j = 0; j < w; j++) {
      double _qx = qx[j];
      double _w = sqrt(_qx * _qx + _qy * _qy) * Simulator2d::R_PARTICLE;
      i_exp[i * w + j] = SQ(3 * (sin(_w) - _w * cos(_w)) / (_w * _w * _w)) *
                         LAUE(Lx * _qx) * LAUE(Ly * _qy);
    }
  }
  return;
}

void gen_initial_config(int n, int La, int Lb, int da, int db,
                        int *a, int *b) {
  int _a = -da, _b = 0;
  for (int i = 0; i < n; i++) {
    _a += da;
    if (_a >= La) {
      _a = 0;
      _b += db;
      if (_b >= Lb) {
        std::cerr << "Error: too many particles" << std::endl;
        exit(1);
      }
    }
    a[i] = _a;
    b[i] = _b;
  }
}

}  // namespace RMC
