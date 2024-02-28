
#include "rmc.hpp"

#include <filesystem>
#include <random>
#include <sstream>
#include <string>

using namespace std;
using namespace Eigen;

#define MOD(x, y) (((x) % (y) + (y)) % (y))
#define REV(x) (MOD((x) + 3, 6))

#define TOROW(out, vec)                                          \
  for (int __i_for_TOROW = 0; __i_for_TOROW < vec.size(); __i_for_TOROW++) { \
    out << vec(__i_for_TOROW) << " ";                                \
  };

# define SLICE(to, from, ini, fin) \
  to.resize((fin) - (ini)); \
  for (int __i_for_SLICE = 0; __i_for_SLICE < (fin) - (ini); __i_for_SLICE++) { \
    to(__i_for_SLICE) = from(__i_for_SLICE + (ini)); \
  };

const int MAX_RAND_ITER = 10000;

/** 0からn-1の整数で埋められたvectorを返す */
vector<int> range(const int n) {
  vector<int> v(n);
  for (int i = 0; i < n; i++) v[i] = i;
  return v;
}

void Simulator::load_exp_data(const string &filename) {
  ifstream fin(filename);
  if (!fin) {
    cerr << "Error: cannot open file `" << filesystem::current_path()
         << filename << "`" << endl;
    exit(1);
  }
  vector<double> q_vec;
  vector<double> i_exp_vec;
  double _q, _i_exp;
  while (fin >> _q >> _i_exp) {
    q_vec.push_back(_q);
    i_exp_vec.push_back(_i_exp);
  }
  fin.close();
  src = filename;

  this->q_exp = Map<VectorXd>(q_vec.data(), q_vec.size());
  this->i_exp = Map<VectorXd>(i_exp_vec.data(), i_exp_vec.size());
  this->i_par.resize(q_exp.size());
  Eigen::VectorXd w = q_exp.array() * R_PARTICLE;
  this->i_par =
      (3 * (sin(w.array()) - w.array() * cos(w.array())) / (w.array().pow(3)))
          .square();
}

void Simulator::load_xtl(const string &filename) {
  ifstream fin(filename);
  if (!fin) {
    cerr << "Error: cannot open file `" << filesystem::current_path()
         << filename << "`" << endl;
    exit(1);
  }
  string line;
  getline(fin, line); // TITLE行
  getline(fin, line); // CELLのみの行

  // 格子定数の読み込み
  getline(fin, line);
  istringstream iss(line);
  double a, b, c, alpha, beta, gamma;
  iss >> a >> b >> c >> alpha >> beta >> gamma;
  int _Lx = (int)(a / (10*A_MG)), _Ly = (int)(b / (10*A_MG));
  assert(abs(alpha - 90) < 1e-6 && abs(beta - 90) < 1e-6 && abs(gamma - 120) < 1e-6);
  assert(abs(_Lx * 10 * A_MG - a) < A_MG  || abs(_Ly * 10 * A_MG - b) < A_MG);
  this->set_Lx(_Lx);
  this->set_Ly(_Ly);

  getline(fin, line); // SYMMETRY NUMBER
  getline(fin, line); // SYMMETRY LABEL
  getline(fin, line); // ATOMS
  getline(fin, line); // NAME X Y Z

  // 粒子位置の読み込み
  vector<int> x_vec(0), y_vec(0);
  while (getline(fin, line)) {
    if (line == "EOF") break;
    istringstream iss(line);
    string name;
    double x, y, z;
    iss >> name >> x >> y >> z;
    int _x = round(Lx * x), _y = round(Ly * y);
    x_vec.push_back(_x);
    y_vec.push_back(_y);
  }
  this->n = x_vec.size();
  this->x = Map<VectorXi>(x_vec.data(), x_vec.size());
  this->y = Map<VectorXi>(y_vec.data(), y_vec.size());
  this->exists = -1 * MatrixXi::Ones(Lx, Ly);
  for (int i = 0; i < n; i++) exists(this->x(i), this->y(i)) = i;

  // x_re, y_re, a_re, a_im, i_simの初期化
  x_re.resize(n);
  y_re.resize(n);
  for (int i = 0; i < n; i++) {
    Vd coord = real_coord(this->x(i), this->y(i));
    x_re(i) = coord.x;
    y_re(i) = coord.y;
  }
  assert(q_exp.size() > 0 || "exp data not loaded");
  compute_i();
}

void Simulator::set_q_range(const double _q_min, const double _q_max) {
  assert(this->q_exp.size() > 0 && this->i_exp.size() == this->q_exp.size() &&
         this->i_par.size() == this->q_exp.size());
  assert(0 <= _q_min && _q_min < _q_max);
  int _q_min_idx = -1, _q_max_idx = -1;
  for (int i = 0; i < q_exp.size(); i++) {
    if (_q_min_idx == -1 && q_exp(i) >= _q_min) {
      _q_min_idx = i;
    }
    if (q_exp(i) <= _q_max) {
      _q_max_idx = i;
    }
  }
  assert(_q_min_idx != -1 && _q_max_idx != -1);

  this->q_min = _q_min;
  this->q_max = _q_max;
  Eigen::VectorXd _q_exp = this->q_exp;
  Eigen::VectorXd _i_exp = this->i_exp;
  Eigen::VectorXd _i_par = this->i_par;
  SLICE(this->i_exp, _i_exp, _q_min_idx, _q_max_idx);
  SLICE(this->i_par, _i_par, _q_min_idx, _q_max_idx);
  SLICE(this->q_exp, _q_exp, _q_min_idx, _q_max_idx);
}

void Simulator::run(const int _max_iter, const double _res_thresh,
                    const double _sigma2, const int move_per_step) {
  this->max_iter = _max_iter;
  this->res_thresh = _res_thresh;
  this->sigma2 = _sigma2;
  this->move_per_step = move_per_step;

  residual = compute_residual();
  residual_hist.resize(max_iter + 1);
  residual_hist(0) = residual;

  for (int i = 0; i < max_iter; i++) {
    step_forword(move_per_step);
    assert(0 <= i + 1 && i + 1 <= max_iter);
    assert(residual_hist.size() == max_iter + 1);
    residual_hist(i + 1) = residual;
    if (residual < res_thresh) {
      residual_hist = residual_hist.segment(0, i + 1);
      break;
    }
  }
}

void Simulator::init() {
  // x, y, existsの初期化
  x.resize(n);
  y.resize(n);
  exists.resize(Lx, Ly);
  for (int _x = 0; _x < Lx; _x++) {
    for (int _y = 0; _y < Ly; _y++) {
      exists(_x, _y) = -1;
    }
  }

  bool failed = 0;
  vector<int> x_shuffled = range(Lx);
  vector<int> y_shuffled = range(Ly);
  for (int i = 0; i < n; i++) {
    this->shuffle(x_shuffled);
    bool success_x = 0;
    for (int _x : x_shuffled) {
      this->shuffle(y_shuffled);
      bool success_y = 0;
      for (int _y : y_shuffled) {
        bool is_clear = 1;
        for (auto &p : PROHIBITED) {
          int __x = MOD(_x + p.x, Lx);
          int __y = MOD(_y + p.y, Ly);
          if (exists(__x, __y) > -1) {
            is_clear = 0;
            break;
          }
        }
        if (!is_clear) continue;  // y
        x(i) = _x;
        y(i) = _y;
        exists(_x, _y) = i;
        success_y = 1;
        break;
      }
      if (success_y) {
        success_x = 1;
        break;
      } else {
        continue;  // x
      }
    }
    if (!success_x) {
      failed = 1;
      break;
    } else {
      continue;  // i
    }
  }

  if (failed) {
    cerr << "Error: failed to initialize particles" << endl;
    exit(1);
  }

  // x_re, y_re, a_re, a_im, i_simの初期化
  x_re.resize(n);
  y_re.resize(n);
  for (int i = 0; i < n; i++) {
    Vd coord = real_coord(x(i), y(i));
    x_re(i) = coord.x;
    y_re(i) = coord.y;
  }
  assert(q_exp.size() > 0 || "exp data not loaded");
  compute_i();

  cout << "Initialization done" << endl;
  residual = compute_residual();
  cout << "initial residual: " << residual << endl;
}

int Simulator::randint(const int max, vector<int> &exclude, const int n_exclude) {
  assert(max > 0);
  if (n_exclude == max) return -1;
  int r;
  auto end = exclude.begin() + n_exclude;
  do {
    r = engine() % max;
  } while (find(exclude.begin(), end, r) != end);
  return r;
}

bool Simulator::try_move(const int i, const int d) {
  assert(0 <= i && i < n);
  assert(0 <= d && d < 6);
  Vi before(x(i), y(i));
  Vi after(MOD(x(i) + STEP[d].x, Lx), MOD(y(i) + STEP[d].y, Ly));
  assert(exists(before.x, before.y) == i);
  assert(exists(after.x, after.y) == -1);

  for (auto &p : PROHIBITED) {
    int __x = MOD(after.x + p.x, Lx);
    int __y = MOD(after.y + p.y, Ly);
    if (exists(__x, __y) == i) continue;
    if (exists(__x, __y) > -1) return 0;
  }

  x(i) = after.x;
  y(i) = after.y;
  exists(before.x, before.y) = -1;
  exists(after.x, after.y) = i;

  Vd coord = real_coord(x(i), y(i));
  x_re(i) = coord.x;
  y_re(i) = coord.y;
  return 1;
}

void Simulator::compute_i() {
  a_re.resize(q_exp.size(), N_THETA);
  a_im.resize(q_exp.size(), N_THETA);
  i_sim.resize(q_exp.size());
  for (int _qi = 0; _qi < q_exp.size(); _qi++) {
    double _q = q_exp(_qi);
    for (int _theta = 0; _theta < N_THETA; _theta++) {
      double _theta_rad = _theta * D_THETA;
      double _qx = _q * cos(_theta_rad), _qy = _q * sin(_theta_rad);
      VectorXd re(n), im(n);
      re = cos(-(_qx * x_re.array() + _qy * y_re.array()));
      im = sin(-(_qx * x_re.array() + _qy * y_re.array()));
      a_re(_qi, _theta) = re.sum();
      a_im(_qi, _theta) = im.sum();
    }
  }

  i_sim = i_par.array()
          * ( a_re.array().square().rowwise().sum()
              + a_im.array().square().rowwise().sum()
          ) * D_THETA / (2 * M_PI);
  i_sim *= i_exp.sum() / i_sim.sum();
}

void Simulator::update_i(const Vd &before, const Vd &after) {
  // 複素振幅の更新
  for (int _qi = 0; _qi < q_exp.size(); _qi++) {
    double _q = q_exp(_qi);
    for (int _theta = 0; _theta < N_THETA; _theta++) {
      double _theta_rad = _theta * D_THETA;
      double _qx = _q * cos(_theta_rad), _qy = _q * sin(_theta_rad);
      assert(0 <= _qi && _qi < q_exp.size());
      assert(0 <= _theta && _theta < N_THETA);
      assert(a_re.rows() == q_exp.size() && a_re.cols() == N_THETA);
      assert(a_im.rows() == q_exp.size() && a_im.cols() == N_THETA);
      a_re(_qi, _theta) += -cos(-(_qx * before.x + _qy * before.y)) +
                           cos(-(_qx * after.x + _qy * after.y));
      a_im(_qi, _theta) += -sin(-(_qx * before.x + _qy * before.y)) +
                           sin(-(_qx * after.x + _qy * after.y));
    }
  }

  // 散乱強度の更新
  i_sim = i_par.array()
          * ( a_re.array().square().rowwise().sum()
              + a_im.array().square().rowwise().sum()
          ) * D_THETA / (2 * M_PI);
  i_sim *= i_exp.sum() / i_sim.sum();
}

double Simulator::compute_residual() const {
  // 和で正規化されていることを確認
  assert(abs(i_sim.sum() / i_exp.sum() - 1) < 1e-6);
  return (i_sim - i_exp).array().square().sum();
}

void Simulator::step_forword(const int n_move) {
  // 評価で棄却された場合に戻すためにコピーを取っておく(値渡し)
  MatrixXd before_a_re = a_re;
  MatrixXd before_a_im = a_im;
  VectorXd before_i_sim = i_sim;
  vector<int> move_i_hist(n_move);
  vector<int> move_d_hist(n_move);

  for (int _n_move = 0; _n_move < n_move; _n_move++) {
    bool move_success = false;
    int n_exclude = 0;
    vector<int> exclude(n);
    for (int rand_iter=0; rand_iter < MAX_RAND_ITER; rand_iter++) {
      int i = randint(n, exclude, n_exclude);
      if (i == -1) {
        cerr << "Error: failed to move any particles" << endl;
        exit(1);
      }
      if (move_success) break;
      vector<int> d_shuffled = range(6);
      this->shuffle(d_shuffled);
      for (int d : d_shuffled) {
        Vd before(x_re(i), y_re(i));
        move_success = try_move(i, d);
        if (!move_success) continue;
        Vd after(x_re(i), y_re(i));
        update_i(before, after);
        // 後ろから埋めて前から戻せば元通りになる
        move_i_hist[n_move - 1 - _n_move] = i;
        move_d_hist[n_move - 1 - _n_move] = d;
        break;
      }
      if (move_success) break;
    }
    if (!move_success) {
      cerr << "Error: failed to move any particles" << endl;
      exit(1);
    }
  }

  double after_residual = compute_residual();
  bool accept =
      (after_residual < residual) ||
      (exp(-(after_residual - residual) / sigma2) > dist(engine));
  if (accept) {
    residual = after_residual;
  } else {
    for (int i = 0; i < n_move; i++) {
      bool move_success = try_move(move_i_hist[i], REV(move_d_hist[i]));
      assert(move_success);
    }
    a_re = before_a_re;
    a_im = before_a_im;
    i_sim = before_i_sim;
  }
}

void Simulator::save_result(const string &filename) const {
  // クラスタ位置をxtlファイルに保存
  string xtl = filename + ".xtl";
  ofstream f_xtl(filename + ".xtl");
  if (!f_xtl) {
    cerr << "Error: cannot open file " << xtl << endl;
    exit(1);
  }
  f_xtl << "TITLE " << filename << endl;
  // xtlの格子サイズはAA単位
  f_xtl << "CELL" << endl
        << "  " << A_MG*Lx*10 << " " << A_MG*Ly*10 << " " << C_MG*10 << " 90 90 120" << endl;
  f_xtl << "SYMMETRY NUMBER 1" << endl;
  f_xtl << "SYMMETRY LABEL  P1" << endl;
  f_xtl << "ATOMS" << endl;
  f_xtl << "NAME  X  Y  Z" << endl;
  f_xtl << setprecision(10) << fixed;
  for (int i = 0; i < n; i++) {
    f_xtl << "L " << (double)x(i)/Lx << " " << (double)y(i)/Ly << " " << 0 << endl;
  }
  f_xtl << "EOF" << endl;
  f_xtl.close();

  // 格子座標でも保存
  string lattice = filename + "_lattice.dat";
  ofstream f_lattice(lattice);
  if (!f_lattice) {
    cerr << "Error: cannot open file " << lattice << endl;
    exit(1);
  }
  f_lattice << ">>> Lx:" << Lx << ", Ly:" << Ly << ", N:" << n;
  f_lattice << " x=[A_MG,0], y=[-A_MG/2, A_MG*sqrt(3)/2], A_MG:" << A_MG;
  f_lattice << " expfile:" << src << endl;
  f_lattice << x.transpose() << endl;
  f_lattice << y.transpose() << endl;

  // 散乱強度をファイルに保存
  string iq = filename + "_iq.dat";
  ofstream f_iq(iq);
  if (!f_iq) {
    cerr << "Error: cannot open file " << iq << endl;
    exit(1);
  }
  for (int _qi = 0; _qi < q_exp.size(); _qi++) {
    f_iq << q_exp(_qi) << " " << i_sim(_qi) << endl;
  }
  f_iq.close();

  // residualの履歴をファイルに保存
  string res = filename + "_residual.dat";
  ofstream f_res(res);
  if (!f_res) {
    cerr << "Error: cannot open file " << res << endl;
    exit(1);
  }
  for (int i = 0; i < residual_hist.size(); i++) {
    f_res << i << " " << residual_hist[i] << endl;
  }

  // パラメータをファイルに保存
  string param = filename + "_param.yaml";
  ofstream f_param(param);
  if (!f_param) {
    cerr << "Error: cannot open file " << param << endl;
    exit(1);
  }
  f_param << "Lx: " << Lx << endl;
  f_param << "Ly: " << Ly << endl;
  f_param << "n: " << n << endl;
  f_param << "A_MG: " << A_MG << endl;
  f_param << "R_PARTICLE: " << R_PARTICLE << endl;
  f_param << "N_THETA: " << N_THETA << endl;
  f_param << "max_iter: " << max_iter << endl;
  f_param << "n_iter_done: " << residual_hist.size() - 1 << endl;
  f_param << "res_thresh: " << res_thresh << endl;
  f_param << "sigma2: " << sigma2 << endl;
  f_param << "move_per_step: " << move_per_step << endl;
  f_param << "q_min: " << q_min << endl;
  f_param << "q_max: " << q_max << endl;
  f_param << "src_file: " << src << endl;
  f_param << "mt19937_seed: " << seed << endl;
  f_param << endl;
  f_param << "q[nm^-1]: [ " << q_exp.transpose() << "]" << endl;
  f_param << "i_exp: [ " << i_exp.transpose() << "]" << endl;

  f_param.close();
}