#include <iostream>
#include <fstream>
#include <random>

using namespace std;

int main()
{
  default_random_engine generator;
  uniform_real_distribution<double> distribution(-1.0, 1.0);

  ofstream ofs("data.txt");
  cout << "input number of train samples of cos funciton: ";
  int num;
  cin >> num;
  ofs << num << endl;
  double x1, y1;
  for(int i = 0; i < num; ++i){
    x1 = distribution(generator);
    y1 = cos(1.57 * x1);
    ofs << x1 << " " << y1 << endl;
  }
  ofs.close();

  ofs.open("test.txt");
  cout << "input number of test samples of cos funciton: ";
  cin >> num;
  ofs << num << endl;
  for(int i = 0; i < num; ++i){
    x1 = distribution(generator);
    y1 = cos(1.57 * x1);
    ofs << x1 << " " << y1 << endl;
  }
  ofs.close();
}
