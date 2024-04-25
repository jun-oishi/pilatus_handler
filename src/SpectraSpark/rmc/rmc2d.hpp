#ifndef RMC2D_HPP
#define RMC2D_HPP

#include <random>

namespace RMC {

using ulong = uint32_t;

#ifdef NAN
#undef NAN
#endif
const double NAN = std::nan("");

class Simulator2d {
 public:
  static constexpr double A_MG = 0.321;
  static constexpr double R_PARTICLE = 0.355;
  static constexpr int N_PROHIBITED = 37;
  static const int STEP_A[6];
  static const int STEP_B[6];
  static const int PROHIBITED_A_REL[N_PROHIBITED];
  static const int PROHIBITED_B_REL[N_PROHIBITED];
  static int max_rand_trial;

  /**
   * @brief Construct a new Simulator2d object
   * @details initialize the random number generator with the given seed
   *          and set the rotation angle to 0
   */
  Simulator2d(const int _seed = 0);
  ~Simulator2d();

  /**
   * @brief Set the model
   * @details set model space dimension and initial configuration of particles
   * @param _La a dimension of the model space
   * @param _Lb b dimension of the model space
   * @param _n number of particles
   * @param _a array of a coordinates of particles
   * @param _b array of b coordinates of particles
   */
  void set_model(int _La, int _Lb, int _n, const int *a, const int *b);

  /**
   * @brief Set the experimental data
   * @details set the experimental data to fit
   * @param w width of the experimental data
   * @param h height of the experimental data
   * @param _qx array of q_x
   * @param _qy array of q_y
   * @param i_exp array of the experimental intensity, row-major order
   */
  void set_exp_data(int w, int h, double *_qx, const double *_qy, const double *i_exp);

  /**
   * @brief move particles without fitting
   * @details Move particles without updating the intensity
   * @throw runtime_error : if random move failed
   * @param n_iter number of iterations
   * @param n_move number of moves per iteration
   */
  void anneal(int n_iter, int n_move=1);

  /**
   * @brief set lattice axis
   * @details assuming a-axis to the given angle and b-axis to 120 degrees from
   * a-axis, setup the transformation matrix from a-b to x-y. angle is measured
   * counter-clockwise from the x-axis
   * @param _theta angle in degrees
   */
  void set_rotation(double _theta);

  /**
   * @brief compute the intensity
   * @details compute the intensity of the model and store it in i_sim
   */
  void compute_i();

  /**
   * @brief run the simulation and return number of steps taken
   * @details run the simulation with the given number of moves and iterations
   *          and store the residual history in res_hist
   * @throw runtime_error : if random move failed
   * @param n_move number of moves per iteration
   * @param max_iter number of iterations
   * @param sigma2 variance of the noise
   * @param thresh threshold to stop the simulation
   * @param res_hist array to store the residual history, memory must be
   * allocated if threshold achieved or aborted, the remaining elements are set
   * to -1
   */
  int run(int n_move, int max_iter, double *res_hist,
           double sigma2=1e-1, double thresh=1e-10);

  /**
   * @brief get the simulated intensity
   *
   * @param w width of the intensity
   * @param h height of the intensity
   * @param qx array to save the qx values
   * @param qy array to save the qy values
   * @param i_sim array to save the simulated intensity
   */
  inline void get_i_sim(int &w, int &h, double *&qx, double *&qy,
                        double *&i_sim) const {
    delete[] qx;
    delete[] qy;
    delete[] i_sim;
    w = this->width;
    h = this->height;
    qx = new double[w], qy = new double[h];
    for (ulong i = 0; i < w; i++) qx[i] = this->qx_seq[i];
    for (ulong i = 0; i < h; i++) qy[i] = this->qy_seq[i];
    ulong n_all = w * h;
    i_sim = new double[n_all];
    for (ulong i = 0; i < n_all; i++) {
      ulong idx = this->map[i];
      if (idx != -1) {
        i_sim[i] = this->i_sim[idx];
      } else {
        i_sim[i] = NAN;
      }
    }
  };

  inline int get_seed() const { return this->seed; };

  inline int get_LA() const { return this->La; };
  inline int get_LB() const { return this->Lb; };
  inline int get_n() const { return this->n; };
  /**
   * @brief get the configuration
   *
   * @param n number of particles
   * @param La a dimension of the model space
   * @param Lb b dimension of the model space
   * @param a array of a coordinates of particles
   * @param b array of b coordinates of particles
   */
  inline void get_config(int &n, int &La, int &Lb, int *&a, int *&b) const {
    n = this->n;
    La = this->La;
    Lb = this->Lb;
    delete[] a;
    delete[] b;
    a = new int[n], b = new int[n];
    for (int i=0; i<n; i++) a[i] = this->a[i], b[i] = this->b[i];
    return;
  };

 private:
  const int seed;
  std::mt19937 engine;
  std::uniform_real_distribution<> dist;

  int La, Lb;
  int n;
  int *a, *b;
  // [i*N_PROHIBITED+j] : i-th particle's j-th relative position
  int *prohibited_a, *prohibited_b;
  double theta; // degree
  double mat_ab2xy[2][2];
  double *x, *y;
  ulong width, height, n_px;
  double *qx_seq, *qy_seq; // 2次元グリッドを定義するqx,qyの配列
  ulong *map; // map[y*width+x] = {画素(x,y)に対応するi_expなどのインデックス, ない場合は-1}
  double *a_re, *a_im; // mapで指定される画素の複素振幅
  double *qx, *qy;     // mapで指定される画素のqx, qy
  double *i_exp, *i_sim, *i_par; // mapで指定される画素の散乱強度
  double i_sum, isq_sum;

  /**
   * @brief move particles
   * @details move particles without updating xy nor intensity
   * @throw runtime_error : if random move failed
   * @param n_move number of moves
   * @param moved_idx array to save moved particle index
   * @param a_before array to save a before move
   * @param b_before array to save b before move
   * @param x_before array to save x before move
   * @param y_before array to save y before move
   * @param x_after array to save x after move
   * @param y_after array to save y after move
  */
  void move(int n_move, int *moved_idx, int *a_before, int *b_before,
            double *x_before, double *y_before, double *x_after, double *y_after);

  /**
   * @brief update prohibited positions in cutoff distance
   * @param idx array of moved particle index
   * @param a_after array of a after move
   * @param b_after array of b after move
   */
  inline void update_prohibited(int idx, int a_after, int b_after) {
    for (int i = 0; i < N_PROHIBITED; i++) {
      this->prohibited_a[idx * N_PROHIBITED + i] =
          (a_after + PROHIBITED_A_REL[i] + La) % La;
      this->prohibited_b[idx * N_PROHIBITED + i] =
          (b_after + PROHIBITED_B_REL[i] + Lb) % Lb;
    }
  };

  /**
   * @brief convert array a-b to x-y
   * @param n number of particles
   * @param a array of a coordinates
   * @param b array of b coordinates
   * @param x array to save x coordinates
   * @param y array to save y coordinates
  */
  inline void ab2xy(int n, const int a[], const int b[], double *x,
                    double *y) const {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      x[i] = a[i] * this->mat_ab2xy[0][0] + b[i] * this->mat_ab2xy[1][0];
      y[i] = a[i] * this->mat_ab2xy[0][1] + b[i] * this->mat_ab2xy[1][1];
    }
  };

  /**
   * @brief update intensity
   * @details update the intensity using stored complex amplitudes and moved particles
   * @param x_before array of x before move
   * @param y_before array of y before move
   * @param x_after array of x after move
   * @param y_after array of y after move
   */
  void update_i(int n, double *x_before, double *y_before, double *x_after, double *y_after);

  /**
   * @brief compute the residual
   * @details compute the residual between the experimental intensity and the
   * simulated intensity and normalize with the sum of the squared experimental
   * intensity
   * @return double residual
   */
  double compute_residual() const;
};


}  // namespace RMC

#endif  // RMC2D_HPP