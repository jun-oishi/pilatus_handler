
#include "rmc.cpp"
#include <iostream>
#include <string>
#include <chrono>

using namespace std;

chrono::system_clock::time_point start;
string ini_xtl = "ini.xtl";

void show_time(const string &msg = "") {
  chrono::system_clock::time_point now = chrono::system_clock::now();
  double elapsed = chrono::duration_cast<chrono::milliseconds>(now - start).count() / 1000.0;
  cout << msg << " " << elapsed << " sec" << endl;
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    cout << "Usage: " << argv[0] << " <expdata> <dst> <n_steps> <sigma2>" << endl;
    cout << "  expdata: path to I(q) data file to fit" << endl;
    cout << "  dst    : path to save result" << endl;
    cout << "  n_steps: number of steps to run" << endl;
    cout << "  sigma2 : variance of the noise" << endl;
    cout << "  note   : ini.xtl is required in the current directory" << endl;
    return 1;
  }

  string expdata = argv[1];
  string dst = argv[2];

  ofstream _result(dst+".xtl");
  if (!_result) {
    cout << "Error: cannot save to " << dst << endl;
    return 1;
  }
  _result.close();

  int n_steps = stoi(argv[3]);
  double sigma2 = stod(argv[4]);

  start = chrono::system_clock::now();

  Simulator sim;
  sim.load_exp_data(expdata);
  show_time("load_exp_data");

  // sim.set_Lx(90);
  // sim.set_Ly(90);
  // sim.set_n(465);
  sim.set_q_range(4.0, 7.0);
  // sim.init();
  sim.load_xtl("ini.xtl");
  show_time("initialize");

  sim.run(n_steps, 1, sigma2, 3);
  show_time("run");

  sim.save_result(dst);
}