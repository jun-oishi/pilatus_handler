#include "rmc2d.hpp"
#include <iostream>
#include <omp.h>
#include <cmath>
#include <set>
#include <random>
#include <algorithm>
#include <cassert>

#define SQ(x) ((x)*(x))

namespace RMC {

const int Simulator2d::STEP_A[] = {1, 0, -1, -1, 0, 1};
const int Simulator2d::STEP_B[] = {0, 1, 1, 0, -1, -1};
const int Simulator2d::PROHIBITED_A_REL[] = {
    0, 1, 1,  0,  -1, -1, 0,                     // 1NN
    2, 1, -1, -2, -1, 1,                         // 2NN
    2, 2, 0,  -2, -2, 0,                         // 3NN
    3, 3, 2,  1,  -1, -2, -3, -3, -2, -1, 1, 2,  // 4NN
    3, 3, 0,  -3, -3, 0                          // 5NN
};
const int Simulator2d::PROHIBITED_B_REL[] = {
    0, 0, 1, 1,  0,  -1, -1,                      // 1NN
    1, 2, 1, -1, -2, -1,                          // 2NN
    0, 2, 2, 0,  -2, -2,                          // 3NN
    1, 2, 3, 3,  2,  1,  -1, -2, -3, -3, -2, -1,  // 4NN
    0, 3, 3, 0,  -3, -3                           // 5NN
};
int Simulator2d::max_rand_trial = 10000;

Simulator2d::Simulator2d(const int _seed) : seed(_seed), engine(_seed), dist(0, 1) {
  this->set_rotation(0);
}

Simulator2d::~Simulator2d() {
  delete[] a;
  delete[] b;
  delete[] prohibited_a;
  delete[] prohibited_b;
  delete[] x;
  delete[] y;
  delete[] a_re;
  delete[] a_im;
  delete[] qx;
  delete[] qy;
  delete[] i_exp;
  delete[] i_sim;
  delete[] i_par;
}

void Simulator2d::set_model(int _La, int _Lb, int _n, const int *_a, const int *_b) {
  this->La = _La;
  this->Lb = _Lb;
  this->n = _n;
  this->a = new int[n];
  this->b = new int[n];
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    this->a[i] = _a[i];
    this->b[i] = _b[i];
  }

  this->prohibited_a = new int[n*N_PROHIBITED];
  this->prohibited_b = new int[n*N_PROHIBITED];
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    this->update_prohibited(i, this->a[i], this->b[i]);
  }

  this->x = new double[n];
  this->y = new double[n];
  this->ab2xy(n, this->a, this->b, this->x, this->y);
}

void Simulator2d::set_exp_data(
  int w, int h, double *_qx, const double *_qy, const double *_i_exp
) {
  this->width = w, this->height = h;
  this->qx_seq = new double[w], this->qy_seq = new double[h];
  for (int i=0; i<w; i++) this->qx_seq[i] = _qx[i];
  for (int i=0; i<h; i++) this->qy_seq[i] = _qy[i];
  ulong n_all = w * h;
  this->map = new ulong[n_all];
  for (ulong i = 0; i < n_all; i++) {
    this->map[i] = -1;
  }
  ulong cur = 0;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      int idx = y * w + x;
      if (std::isfinite(_i_exp[idx])) {
        double _q2 = _qx[x] * _qx[x] + _qy[y] * _qy[y];
        if (_q2 < 1e-10) continue;
        this->map[y*w + x] = cur;
        cur++;
      }
    }
  }
  this->n_px = cur;

  this->qx = new double[this->n_px];
  this->qy = new double[this->n_px];
  this->i_exp = new double[this->n_px];
  this->i_sim = new double[this->n_px];
  this->a_re = new double[this->n_px];
  this->a_im = new double[this->n_px];
  this->i_par = new double[this->n_px];
  this->i_sum = 0, this->isq_sum = 0;
  cur = 0;
  for (ulong i = 0; i < n_all; i++) {
    int y = i / w, x = i % w;
    if (this->map[i] == -1) continue;
    ulong xy = y * w + x;
    this->qx[cur] = _qx[x], this->qy[cur] = _qy[y];
    this->i_exp[cur] = _i_exp[xy];
    this->i_sum += _i_exp[xy];
    this->isq_sum += SQ(_i_exp[xy]);
    double _w = sqrt(_qx[x] * _qx[x] + _qy[y] * _qy[y]) * this->R_PARTICLE;
    this->i_par[cur] = SQ(3 * (sin(_w) - _w * cos(_w)) / (_w * _w * _w));
    cur++;
  }
}

void Simulator2d::anneal(int n_iter, int n_move) {
  int *idx = new int[n_move];
  int *a_before = new int[n_move];
  int *b_before = new int[n_move];
  double *_d = new double[n_move];
  int i;
  try {
    for (i=0; i<n_iter; i++) {
      this->move(n_move, idx, a_before, b_before, _d, _d, _d, _d);
    }
  } catch (const std::runtime_error e) {
    std::cerr << e.what() << " at step " << i << std::endl;
  }
  this->ab2xy(n, this->a, this->b, this->x, this->y);
  delete[] idx;
  delete[] a_before;
  delete[] b_before;
  delete[] _d;
}

void Simulator2d::set_rotation(double _theta) {
  this->theta = _theta;
  _theta = _theta * M_PI / 180;
  double eax = cos(_theta), eay = sin(_theta);
  _theta += 2 * M_PI / 3;
  double ebx = cos(_theta), eby = sin(_theta);
  this->mat_ab2xy[0][0] = eax * this->A_MG;
  this->mat_ab2xy[0][1] = eay * this->A_MG;
  this->mat_ab2xy[1][0] = ebx * this->A_MG;
  this->mat_ab2xy[1][1] = eby * this->A_MG;
}

void Simulator2d::compute_i() {
  double _i_sum = 0;
  #pragma omp parallel for reduction(+:_i_sum)
  for (ulong i = 0; i < this->n_px; i++) {
    a_re[i] = 0;
    a_im[i] = 0;
    for (int j = 0; j < this->n; j++) {
      double delta = qx[i] * x[j] + qy[i] * y[j];
      a_re[i] += cos(delta);
      a_im[i] += sin(delta);
    }
    i_sim[i] = i_par[i] * (SQ(a_re[i]) + SQ(a_im[i]));
    _i_sum += i_sim[i];
  }

  double c = this->i_sum / _i_sum;
  #pragma omp parallel for
  for (ulong i = 0; i < this->n_px; i++) {
    i_sim[i] *= c;
  }
}

double Simulator2d::compute_residual() const {
  double residual = 0;
  #pragma omp parallel for reduction(+:residual)
  for (ulong i = 0; i < this->n_px; i++) {
    residual += SQ(i_exp[i] - i_sim[i]);
  }
  return residual / isq_sum;
}

int Simulator2d::run (
  int n_move, int max_iter, double *res_hist, double sigma2, double thresh
) {
  this->compute_i();
  res_hist[0] = this->compute_residual();
  double inv_sigma = 1 / sigma2;

  int *idx = new int[n_move];
  int *a_before = new int[n_move];
  int *b_before = new int[n_move];
  double *x_before = new double[n_move];
  double *y_before = new double[n_move];
  double *x_after = new double[n_move];
  double *y_after = new double[n_move];

  int i;
  try {
    for (i=1; i<=max_iter; i++) {
      this->move(n_move, idx, a_before, b_before, x_before, y_before, x_after, y_after);
      this->update_i(n_move, x_before, y_before, x_after, y_after);
      res_hist[i] = this->compute_residual();
      assert (res_hist[i] >= 0);
      if (res_hist[i] < res_hist[i - 1]) {
        // accept
        if (res_hist[i] < thresh) break;
      } else if (dist(engine) <
                exp(-(res_hist[i] - res_hist[i - 1]) * inv_sigma)) {
        // acceptして収束判定
        if (res_hist[i] < thresh) break;
      } else {
        // rejectして元に戻す
        res_hist[i] = res_hist[i-1];
        for (int j=0; j<n_move; j++) {
          this->a[idx[j]] = a_before[j];
          this->b[idx[j]] = b_before[j];
          this->update_prohibited(idx[j], a_before[j], b_before[j]);
        }
      }
    }
  } catch (const std::runtime_error e) {
    std::cerr << e.what() << " at step " << i << std::endl;
  }
  delete[] idx;
  delete[] a_before;
  delete[] b_before;
  delete[] x_before;
  delete[] y_before;
  delete[] x_after;
  delete[] y_after;
  return i;
}

void Simulator2d::move(
  int n_move, int *idx, int *a_before, int *b_before,
  double *x_before, double *y_before, double *x_after, double *y_after
) {
  // Not parallelized to ensure repeatability of random number generation
  std::set<int> idx_set;
  int dir[] = {0, 1, 2, 3, 4, 5};
  ulong n_prohibited = this->n * N_PROHIBITED;
  int all_success = 1;
  for (int trial=0; trial < this->max_rand_trial; trial++) {
    // choose non-repeating indices
    idx_set.clear();
    while (idx_set.size() < (ulong)n_move) {
      int idx_i = this->engine() % this->n;
      if (idx_set.find(idx_i) == idx_set.end()) {
        idx[idx_set.size()] = idx_i;
        idx_set.insert(idx_i);
      }
    }

    // save current position
    for (int i=0; i<n_move; i++) {
      a_before[i] = this->a[idx[i]];
      b_before[i] = this->b[idx[i]];
    }

    for (int i=0; i<n_move; i++) {
      // check if the move is possible
      int particle_success = 0;
      int toa, tob;
      std::shuffle(dir, dir+6, this->engine);
      for (int j=0; j<6; j++) {
        int overlap = 0;
        toa = (this->a[idx[i]] + STEP_A[dir[j]] + La) % La;
        tob = (this->b[idx[i]] + STEP_B[dir[j]] + Lb) % Lb;
        #pragma omp parallel for reduction(+:overlap)
        for (ulong k=0; k < n_prohibited; k++) {
          overlap += (toa == prohibited_a[k] && tob == prohibited_b[k]);
        }
        // overlap==1 means o.k.
        if (overlap == 1) {
          particle_success = 1;
          break;
        } else {
          continue;
        }
      }
      if (particle_success) {
        this->a[idx[i]] = toa;
        this->b[idx[i]] = tob;
        this->update_prohibited(idx[i], toa, tob);
      } else {
        all_success = 0;
        break;
      }
    }

    if (all_success) {
      break; // success
    } else {
      // revert and try again
      for (int i=0; i<n_move; i++) {
        this->a[idx[i]] = a_before[i];
        this->b[idx[i]] = b_before[i];
        this->update_prohibited(idx[i], a_before[i], b_before[i]);
      }
      continue;
    }
  }

  if (!all_success) {
    throw std::runtime_error("Runtime Error : Failed to move particles");
  };

  int *a_after = new int[n_move];
  int *b_after = new int[n_move];
  for (int i=0; i<n_move; i++) {
    int idx_i = idx[i];
    x_before[i] = this->x[idx_i];
    y_before[i] = this->y[idx_i];
    a_after[i] = this->a[idx_i];
    b_after[i] = this->b[idx_i];
  }
  this->ab2xy(n_move, a_after, b_after, x_after, y_after);
  delete[] a_after;
  delete[] b_after;
  return;
}

void Simulator2d::update_i(
  int n, double *x_before, double *y_before, double *x_after, double *y_after
) {
  // ulong n_px = width * height;
  for (int i = 0; i < n; i++) {
    double xb = x_before[i], yb = y_before[i], xa = x_after[i], ya = y_after[i];
    #pragma omp parallel for
    for (ulong j = 0; j < n_px; j++) {
      double delta_b = qx[j] * xb + qy[j] * yb;
      double delta_a = qx[j] * xa + qy[j] * ya;
      this->a_re[j] += -cos(delta_b) + cos(delta_a);
      this->a_im[j] += -sin(delta_b) + sin(delta_a);
    }
  }

  double i_sum = 0;
  #pragma omp parallel for reduction(+:i_sum)
  for (ulong i = 0; i < n_px; i++) {
    this->i_sim[i] = i_par[i] * (SQ(this->a_re[i]) + SQ(this->a_im[i]));
    i_sum += this->i_sim[i];
  }

  double c = this->i_sum / i_sum;
  #pragma omp parallel for
  for (ulong i = 0; i < n_px; i++) {
    this->i_sim[i] *= c;
  }
}

}  // namespace RMC
