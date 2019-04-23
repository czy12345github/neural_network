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

//  net.addOptimizerSGD(0.01);
//  cout << "net addOptimizerSGD" << endl;

  net.addLinearLayer(1, 10);
  cout << "net addLinearLayer" << endl;
  net.addReluActivator();
  cout << "net addReluActivator" << endl;

  net.addLinearLayer(10, 10);
  cout << "net addLinearLayer" << endl;
  net.addReluActivator();
  cout << "net addReluActivator" << endl;

  net.addLinearLayer(10, 1);
  cout << "net addLinearLayer" << endl;
  net.addReluActivator();
  cout << "net addReluActivator" << endl;

  Matrix train_data, test_data;
  int batch;
  read_data("../data/data.txt", train_data);
  batch = read_data("../data/test.txt", test_data);
  cout << "read data" << endl;

  cout << "begin training" << endl;
  net.train(train_data, 50, 10);

  net.eval();
  cout << endl << "test:" << endl;
  Matrix input, output, target;
  test_data.load_data(0, batch, 1, input, target);
  output = net.forward(&input);
  for(int i = 0; i < batch; ++i){
    cout << "output: " << output.val(0, i);
    cout << ", target: " << target.val(0, i) << endl;
  }
}
