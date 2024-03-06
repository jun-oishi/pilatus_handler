
#include "rmc.hpp"
#include <iostream>
#include <string>
#include <chrono>

using namespace std;

chrono::system_clock::time_point start;

void show_time(const string &msg = "") {
  chrono::system_clock::time_point now = chrono::system_clock::now();
  double elapsed = chrono::duration_cast<chrono::milliseconds>(now - start).count() / 1000.0;
  cout << msg << " " << elapsed << " sec" << endl;
}

int main(int argc, char *argv[]) {
  if (argc < 8) {
    cout << "Usage: " << argv[0] << " <expdata> <initial> <dst> <n_steps> <sigma2>" << endl;
    cout << "  expdata: path to I(q) data file to fit" << endl;
    cout << "  initial: path to initial xtl file" << endl;
    cout << "  dst    : path to save result" << endl;
    cout << "  n_steps: number of steps to run" << endl;
    cout << "  sigma2 : variance of the noise" << endl;
    cout << "  q_min  : minimum q to fit" << endl;
    cout << "  q_max  : maximum q to fit" << endl;
    return 1;
  }

  string expdata = argv[1];
  string ini_xtl = argv[2];
  string dst = argv[3];
  int n_steps = stoi(argv[4]);
  double sigma2 = stod(argv[5]);
  double q_min = stod(argv[6]);
  double q_max = stod(argv[7]);

  assert(n_steps >= 100);
  assert(sigma2 > 0);
  assert(0 < q_min && q_min < 10 && q_min < q_max);
  assert(0 < q_max && q_max < 10);

  ofstream _result(dst + ".xtl");
  if (!_result) {
    cout << "Error: cannot save to " << dst << endl;
    return 1;
  }
  _result.close();


  start = chrono::system_clock::now();

  Simulator sim;
  sim.load_exp_data(expdata);
  show_time("load_exp_data");

  sim.set_q_range(q_min, q_max);
  sim.load_xtl(ini_xtl);
  show_time("initialize");

  sim.run(n_steps, 1, sigma2, 3);
  show_time("run");

  sim.save_result(dst);
}