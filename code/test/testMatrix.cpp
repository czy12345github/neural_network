#include <iostream>

#include "../Matrix.h"

using namespace std;

int main()
{
  Matrix ma(2, 2, 1);
  Matrix mb(2, 2, 1);

  ma.val(0, 0) = 1;
  ma.val(0, 1) = 2;
  //ma.val(0, 2) = 3;
  ma.val(1, 0) = 4;
  ma.val(1, 1) = 5;
  //ma.val(1, 2) = 6;

  Matrix mv(2, 1, 1);

  Matrix mc = ma.dot(mb) + mv.dot(Matrix(1, mb.get_col(), 1));;
  for(int i = 0; i < 2; ++i){
    for(int j = 0; j < 2; ++j) cout << mc.val(i, j) << ' ';
    cout << endl;
  }
}
