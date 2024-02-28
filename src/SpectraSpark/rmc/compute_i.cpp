
#include "rmc.cpp"
#include <iostream>
#include <string>

using namespace std;

string EXPDATA = "exp_iq.dat";

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cout << "Usage: " << argv[0] << " <src> <dst>" << endl;
    cout << "  src: path to xtl file" << endl;
    cout << "  dst: path to save result" << endl;
    cout << "  note: exp_iq.dat is required in the current directory" << endl;
    return 1;
  }
  string src = argv[1];
  string dst = argv[2];
  Simulator sim;
  sim.load_exp_data(EXPDATA);
  sim.load_xtl(src);
  sim.run(0, 0, 1, 0);
  sim.save_result(dst);
}