#include <iostream>
#include <fstream>
#include <random>

using namespace std;

int main()
{
  default_random_engine generator;
  uniform_real_distribution<double> distribution(-1.0, 1.0);

  ofstream ofs("data.txt");
  cout << "input number of samples of sin funciton: ";
  int num;
  cin >> num;
  ofs << num << endl;
  int x1, x2, y;
  for(int i = 0; i < num; ++i){
    x1 = distribution(generator) > 0 ? 1 : 0;
    x2 = distribution(generator) > 0 ? 1 : 0;
    y = x1 ^ x2;
    ofs << x1 << " " << x2 << " " << y << endl;
  }
  ofs.close();
}
