#include <iostream>
#include <fstream>
#include <random>

using namespace std;

int main()
{
  ofstream ofs("sin_sequence");
  cout << "input number of train samples of cos funciton: ";
  int num;
  cin >> num;
  ofs << num << endl;
  double step_size;
  cout << "input step size: ";
  cin >> step_size;
  double x = 0.0;
  for(int i = 0; i < num; ++i){
    ofs << sin(x) << " " << sin(x+step_size) << endl;
    x += step_size;
  }
  ofs.close();
}
