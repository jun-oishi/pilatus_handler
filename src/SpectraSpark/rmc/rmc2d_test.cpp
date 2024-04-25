#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <iomanip>
#include "rmc2d.hpp"
#include "rmc2d_util.hpp"

using namespace std;
using namespace RMC;

chrono::system_clock::time_point start;
int WIDTH;
int HEIGHT;
double DQ = 0.08; // nm^-1
int N_PARTICLES = 2500;
int N_STEPS_ANNEAL = 10'000;
int N_STEPS_FIT = 10'000;
int N_MOVE = 1;
int LA = 500;
int LB = 500;
double sigma2 = 1e-3;

void show_time(const string &msg = "") {
  chrono::system_clock::time_point now = chrono::system_clock::now();
  int elapsed = chrono::duration_cast<chrono::milliseconds>(now - start).count();
  cout << msg << " " << elapsed << " ms" << endl;
}

int main(int argc, char *argv[]) {

  int seed;
  if (argc > 1) {
    seed = atoi(argv[1]);
  } else {
    seed = chrono::system_clock::now().time_since_epoch().count();
  }
  // seed = -341653726;

  # ifdef _OPENMP
  cout << "num threads: " << omp_get_max_threads() << endl;
  # endif

  start = chrono::system_clock::now();

  int *a = new int[N_PARTICLES], *b = new int[N_PARTICLES];
  double *qx = nullptr, *qy = nullptr, *i_tmp = nullptr;

  gen_initial_config(N_PARTICLES, LA, LB, 10, 10, a, b);

  // qx = new double[WIDTH], qy = new double[HEIGHT];
  // i_tmp = new double[WIDTH*HEIGHT];
  // for (int i=0; i<WIDTH; i++) qx[i] = (i - WIDTH/2 + 0.5) * DQ;
  // for (int i=0; i<HEIGHT; i++) qy[i] = (i - HEIGHT/2 + 0.5) * DQ;
  // double a_mg = Simulator2d::A_MG;
  // gen_sample_data(WIDTH, HEIGHT, qx, qy, i_tmp, 4*a_mg, 6*a_mg);
  // save_i("i_exp.dat", WIDTH, HEIGHT, qx, qy, i_tmp);

  load_i("i_exp1.dat", WIDTH, HEIGHT, qx, qy, i_tmp);

  show_time("generate data");

  RMC::Simulator2d sim(seed);
  sim.max_rand_trial = 1'000'000;
  cout << "seed: " << sim.get_seed() << endl;
  sim.set_model(LA, LB, N_PARTICLES, a, b);
  sim.set_exp_data(WIDTH, HEIGHT, qx, qy, i_tmp);
  show_time("set model");
  save_config("ini.conf", sim);

  sim.compute_i();
  sim.get_i_sim(WIDTH, HEIGHT, qx, qy, i_tmp);
  show_time("compute i");
  save_i("i_ini.dat", WIDTH, HEIGHT, qx, qy, i_tmp);

  try {
    sim.anneal(N_STEPS_ANNEAL, N_MOVE);
  } catch (const std::runtime_error e) {
    // cout << e.what() << endl;
  }
  show_time("anneal");
  save_config("annealed.conf", sim);

  sim.compute_i();
  sim.get_i_sim(WIDTH, HEIGHT, qx, qy, i_tmp);
  show_time("compute i");
  save_i("i_annealed.dat", WIDTH, HEIGHT, qx, qy, i_tmp);

  double *res_hist = new double[N_STEPS_FIT+1];
  try {
    sim.run(N_MOVE, N_STEPS_FIT, res_hist, sigma2);
  } catch (const std::runtime_error e) {
    // cout << e.what() << endl;
  }
  show_time("run");

  save_result("fit", sim, N_STEPS_FIT, res_hist);
}
