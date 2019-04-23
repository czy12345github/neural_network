#include <iostream>
#include <fstream>
#include <string>

#include "Network.h"
#include "Matrix.h"

using namespace std;

int read_data(string fname, Matrix &data){
  ifstream in(fname);
  int N;
  in >> N;
  data = Matrix(N, 2);
  for(int i = 0; i < N; ++i){
    for(int j = 0; j < 2; ++j){
      in >> data.val(i, j);
    }
  }
  in.close();
  return N;
}

int main(){
  Network net;

  net.addMSELoss();
  cout << "net addMSELoss" << endl;

  net.addOptimizerAdam();
  cout << "net addOptimizerAdam" << endl;

  // net.addOptimizerSGD(0.01);
  // cout << "net addOptimizerSGD" << endl;

  net.addLSTMLayer(1, 51);
  cout << "net addLSTMLayer" << endl;

  net.addLSTMLayer(51, 51);
  cout << "net addLSTMLayer" << endl;

  net.addLinearLayer(51, 1);
  cout << "net addLinearLayer" << endl;
  // net.addReluActivator();
  // cout << "net addReluActivator" << endl;

  Matrix train_data;
  int batch;
  read_data("../data/sin_wave", train_data);
  cout << "read data" << endl;

  cout << "begin training" << endl;
  net.lstmTrain(train_data, 100);

  net.eval();
  cout << endl << "test:" << endl;
}
