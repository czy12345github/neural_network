#ifndef _NETWORK_H
#define _NETWORK_H

#include "Layer.h"
#include "LossFunction.h"
#include "Optimizer.h"

#include <cstdlib>

#include <iostream>

class Network {
public:
  Network(): firstlayer(NULL), lastlayer(NULL), is_training(false) {}
  ~Network(){
    Layer *it = firstlayer;
    while(it != lastlayer){
      firstlayer = firstlayer->nextLayer();
      it->release_resource();
      delete it;
      it = firstlayer;
    }
    delete it;
  }

  void addLinearLayer(int in_dim, int out_dim){
    Layer *newlayer = new LinearLayer(in_dim, out_dim, lastlayer, optim);
    _addLayer(in_dim, out_dim, newlayer);
  }

  void addLSTMLayer(int in_dim, int out_dim){
    Layer *newlayer = new LSTMLayer(in_dim, out_dim, lastlayer, optim);
    _addLayer(in_dim, out_dim, newlayer);
  }

  void addReluActivator(){
    checkLastLayer();

    lastlayer->addActivator(ActivatorFactory::reluActivator());
  }

  // should not use sigmoid activator
  void addSigmoidActivator(){
    checkLastLayer();

    lastlayer->addActivator(ActivatorFactory::sigmoidActivator());
  }

  void addTanhActivator(){
    checkLastLayer();

    lastlayer->addActivator(ActivatorFactory::sigmoidActivator());
  }

  void addMSELoss(){
    loss_func = new MSELoss();
  }

  void addOptimizerSGD(double lr){
    optim = new SGDFactory(lr);
  }

  void addOptimizerAdam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999,
    double eps = 1e-8){

    optim = new AdamFactory(lr, beta1, beta2, eps);
  }

  Matrix forward(const Matrix *in){
    firstlayer->forward(in);
    //std::cout << "firstlayer forward successfully" << std::endl;

    if(firstlayer == lastlayer) return lastlayer->get_output();

    for(auto it = firstlayer->nextLayer(); it != lastlayer; it = it->nextLayer())
      it->forward(NULL);

    //std::cout << "before lastlayer" << std::endl;
    lastlayer->forward(NULL);

    return lastlayer->get_output();
  }

  void backprop(const Matrix *d_out){
    lastlayer->backprop(d_out);

    if(firstlayer == lastlayer) return;

    for(auto it = lastlayer->preLayer(); it != firstlayer; it = it->preLayer())
      it->backprop(NULL);

    firstlayer->backprop(NULL);
  }

  Matrix *parameters(){
    
  }

  void train(Matrix &data, int epochs, int batch_size = 32, bool shuffle = true){
    is_training = true;

    int N = data.get_row();
    int batch;
    Matrix input, target, deriv;
    double loss;

    for(int i = 0; i < epochs; ++i){
      if(shuffle) data.shuffle();
      loss = 0.0;
      for(int k = 0; k < N; k += batch_size){
        batch = (N - k) >= batch_size ? batch_size : (N - k);
        data.load_data(k, batch, input_dim, input, target);
        loss += loss_func->compute_loss(forward(&input), target, deriv);
        // deriv.print_data();
        //std::cout << "deriv row: " << deriv.get_row() << ", col: " << deriv.get_col() << std::endl;
        backprop(&deriv);
      }
      std::cout << "epoch " << (i+1) << ": " << loss << std::endl;
    }
  }

  void lstmTrain(Matrix &data, int epochs){
    Matrix input, target, deriv;
    double loss;

    data.load_data(0, data.get_row(), input_dim, input, target);
    std::cout << "load data successfully" << std::endl;

    for(int i = 0; i < epochs; ++i){
      loss = loss_func->compute_loss(forward(&input), target, deriv);
      //std::cout << "forward successfully" << std::endl;
      //std::cout << "deriv row: " << deriv.get_row() << ", col: " << deriv.get_col() << std::endl;
      backprop(&deriv);

      std::cout << "epoch " << (i+1) << ": " << loss << std::endl;
    }
  }

  void eval(){
    is_training = false;
  }

private:
  Layer *firstlayer;
  Layer *lastlayer;

  int input_dim;
  int output_dim;

  LossFunction *loss_func;

  OptimizerFactory *optim;

  bool is_training;

  void checkLastLayer(){
    if(lastlayer == NULL){
      std::cout << "lastlayer is NULL" << std::endl;
      exit(-1);
    }
  }

  void _addLayer(int in_dim, int out_dim, Layer *newlayer){
    output_dim = out_dim;
    lastlayer = newlayer;

    if(firstlayer == NULL){
      firstlayer = newlayer;
      input_dim = in_dim;
    }
  }
};

#endif
