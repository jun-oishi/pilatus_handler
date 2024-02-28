
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <random>

const double _SQRT3BY2 = sqrt(3) / 2;

template <typename T>
struct Vec {
  T x, y;
  Vec(const T _x, const T _y) : x(_x), y(_y) {};
  Vec() {};

  Vec<T> operator+(const Vec<T> &v) const { return Vec<T>(x + v.x, y + v.y); }

  Vec<T> operator==(const Vec<T> &v) const { return x == v.x && y == v.y; }
};

using Vi = Vec<int>;
using Vd = Vec<double>;

class Simulator {
 public:
  Simulator(const int _seed = 0) : seed(_seed) {
    this->Lx = 0;
    this->Ly = 0;
    this->n = 0;
    this->max_iter = 0;
    this->move_per_step = 0;
    this->res_thresh = std::nan("");
    this->sigma2 = std::nan("");
    this->q_min = std::nan("");
    this->q_max = std::nan("");
    this->x.resize(0);
    this->y.resize(0);
    this->x_re.resize(0);
    this->y_re.resize(0);
    this->a_re.resize(0, 0);
    this->a_im.resize(0, 0);
    this->q_exp.resize(0);
    this->i_exp.resize(0);
    this->i_sim.resize(0);
    this->i_par.resize(0);
    this->residual = std::nan("");
    this->residual_hist.resize(0);
    this->exists.resize(0, 0);
    this->engine = std::mt19937(seed);
    this->dist = std::uniform_real_distribution<double>(0, 1);
  }

  /** Lx, Ly をセットする */
  inline void set_Lx(const int _Lx) {
    this->Lx = _Lx;
    this->Ly = _Lx;
  }

  inline void set_Ly(const int _Ly) { this->Ly = _Ly; }

  /** 粒子数nをセットする*/
  inline void set_n(const int _n) { this->n = _n; }

  /**
   * @brief ファイルからiqの実験値を読み込む
   *
   * @param filename
   *    ヘッダなしのスペース区切りのファイルで1列目がq、2列目がI(q)の実験値
   */
  void load_exp_data(const std::string &filename);

  /**
   * @brief ファイルからxtlファイルを読み込む
   * 格子定数がA_MGの整数倍から1割以上離れていればエラーを出す
   *
   * @param filename
   */
  void load_xtl(const std::string &filename);

  /** iを評価するqの範囲を指定する load_exp_dataの後に実行する */
  void set_q_range(const double _q_min, const double _q_max);

  /** 粒子の初期配置を決定して配列を初期化する */
  void init();

  void run(const int max_iter, const double res_thresh, const double sigma2,
           const int move_per_step);

  /** フィッティング結果を保存する */
  void save_result(const std::string &filename) const;

  inline double get_residual() const { return residual; }

 private:
  std::string src;
  int Lx;
  int Ly;
  int n;
  int max_iter;
  int move_per_step;
  double res_thresh;
  double sigma2;
  double q_min;
  double q_max;
  Eigen::VectorXi x;
  Eigen::VectorXi y;
  Eigen::VectorXd x_re;
  Eigen::VectorXd y_re;
  Eigen::MatrixXd a_re;
  Eigen::MatrixXd a_im;
  Eigen::VectorXd q_exp;
  Eigen::VectorXd i_exp;
  Eigen::VectorXd i_sim;
  Eigen::VectorXd i_par;
  double residual;
  Eigen::VectorXd residual_hist;
  // exists[x,y]にはx,yに粒子が無ければ-1、あればその粒子のインデックスを入れる
  Eigen::MatrixXi exists;
  std::mt19937 engine;
  std::uniform_real_distribution<double> dist;

  const int seed;
  const double A_MG = 0.321;
  const double C_MG = 0.521;
  const double R_PARTICLE = 0.355;
  const std::array<Vi, 6> STEP = {Vi(1, 0),  Vi(1, 1),   Vi(0, 1),
                                  Vi(-1, 0), Vi(-1, -1), Vi(0, -1)};
  const int N_THETA = 180;
  const double D_THETA = 2 * M_PI / N_THETA;
  /** 第5近接以内の禁止区域 */
  const Vi PROHIBITED[37] = {
      Vi(0, 0),  // 0NN
      Vi(1, 0),   Vi(1, 1),   Vi(0, 1),  Vi(-1, 0),
      Vi(-1, -1), Vi(0, -1),  // 1NN
      Vi(2, 0),   Vi(2, 1),   Vi(2, 2),  Vi(1, 2),
      Vi(0, 2),   Vi(-1, 1),  Vi(-2, 0), Vi(-2, -1),
      Vi(-2, -2), Vi(-1, -2), Vi(0, -2), Vi(1, -1),  // 2,3NN
      Vi(3, 0),   Vi(3, 1),   Vi(3, 2),
      Vi(3, 3),   Vi(2, 3),   Vi(1, 3),
      Vi(0, 3),  Vi(-1, 2),  Vi(-2, 1),
      Vi(-3, 0), Vi(-3, -1), Vi(-3, -2),
      Vi(-3, -3), Vi(-2, -3), Vi(-1, -3),
      Vi(0, -3), Vi(1, -2), Vi(2, -1)  // 4,5NN
  };

  inline void shuffle(std::vector<int> &v) { std::shuffle(v.begin(), v.end(), engine); }

  /** 0からmaxまでの乱数を生成する */
  int randint(const int max, std::vector<int> &exclude, const int n_exclude);

  /** 粒子を移動させても禁止域に近づかないか確認してだめなら0を返し可能ならx, y,
   * exists, x_re, y_reを更新する */
  bool try_move(const int i, const int d);

  inline Vd real_coord(const int _x, const int _y) const {
    return Vd((_x - 0.5 * _y) * A_MG, _y * _SQRT3BY2 * A_MG);
  }

  /** 全粒子のxy座標から散乱振幅を計算する */
  void compute_i();

  /** a_re, a_imを更新して使って散乱強度を再計算する */
  void update_i(const Vd &old, const Vd &now);

  /** residualを計算する */
  double compute_residual() const;

  /** 粒子の配置をランダムに変更してiを評価、変更を採用または棄却して各変数を更新する */
  void step_forword(const int n_move);
};