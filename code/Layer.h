#ifndef _LAYER_H
#define _LAYER_H

#include <cmath>

#include "Matrix.h"
#include "Optimizer.h"

class Activator {
public:
  Activator(double (*activate_func)(double), double (*derivate_func)(double)):
  activate_func(activate_func), derivate_func(derivate_func) {}

  Matrix activate(const Matrix &in){
    return in.call_func(activate_func);
  }

  Matrix derivate(const Matrix &in){
    return in.call_func(derivate_func);
  }

private:
  double (*activate_func)(double);

  // the input is the activated value
  double (*derivate_func)(double);
};

class ActivatorFactory {
public:
  static Activator *reluActivator(){
    return new Activator(relu_activate_func, relu_derivate_func);
  }

  static Activator *sigmoidActivator(){
    return new Activator(sigmoid_activate_func, sigmoid_derivate_func);
  }

  static Activator *tanhActivator(){
    return new Activator(tanh, tanh_derivate_func);
  }

  static double relu_activate_func(double x){
    if(x > 0) return x;
    return 0.01 * x;
  }

  static double relu_derivate_func(double x){
    if(x > 0) return 1.0;
    return 0.01;
  }

  static double sigmoid_activate_func(double x){
    return 1.0 / (1.0 + exp(-x));
  }

  static double sigmoid_derivate_func(double x){
    return x * (1.0 - x);
  }

  static double tanh_derivate_func(double x){
    return (1.0 - x * x);
  }
};

class Layer {
public:
  Layer(Layer *pre, int params_row, int params_col):
  params(params_row, params_col), d_params(params_row, params_col),
  pre_layer(pre), next_layer(NULL), in(NULL), act(NULL) {
    if(pre != NULL){
      pre->next_layer = this;
      is_first_layer = false;
    }
    else is_first_layer = true;
  }

  Matrix get_output(){
    return out;
  }

  Layer *nextLayer(){
    return next_layer;
  }

  Layer *preLayer(){
    return pre_layer;
  }

  void addActivator(Activator *activator){
    act = activator;
  }

  virtual void forward(const Matrix *) = 0;

  virtual void backprop(const Matrix *) = 0;

  virtual void release_resource() {}

protected:
  Matrix d_in;
  Matrix out;
  const Matrix *in;
  const Matrix *d_out;

  Layer *pre_layer;
  Layer *next_layer;

  Activator *act;

  Matrix params, d_params;

  bool is_first_layer;
};


class LSTMLayer: public Layer {
public:
  LSTMLayer(int, int, Layer *, OptimizerFactory *);

  void forward(const Matrix *) override;

  void backprop(const Matrix *) override;

  void release_resource() override;

private:
  int in_dim, out_dim;

  Matrix wgx, wgh, bg; // the tanh layer
  Matrix wix, wih, bi; // input gate
  Matrix wfx, wfh, bf; // forget gate
  Matrix wox, woh, bo; // output gate

  Matrix d_wgx, d_wgh, d_bg;
  Matrix d_wix, d_wih, d_bi;
  Matrix d_wfx, d_wfh, d_bf;
  Matrix d_wox, d_woh, d_bo;

  /*
  state and value of each time step, each saved as a row.
  at last, transpose h to get out
  */
  Matrix state, h;

  // save gt, it, ft, and ot of each time step
  Matrix seq_g, seq_i, seq_f, seq_o;

  Activator *act_tanh, *act_sigmoid;

  Optimizer *optim;

  void clear_matrix(){
    state = Matrix(1, wgx.get_row());
    h = Matrix(1, wgx.get_row());
    seq_g = Matrix();
    seq_i = Matrix();
    seq_f = Matrix();
    seq_o = Matrix();
  }

  void reset_params(Matrix &);

  void init_params(){
    params.set_vals(0, out_dim, 0, in_dim, wgx);
    params.set_vals(0, out_dim, in_dim, in_dim + out_dim, wgh);

    params.set_vals(out_dim, 2*out_dim, 0, in_dim, wix);
    params.set_vals(out_dim, 2*out_dim, in_dim, in_dim + out_dim, wih);

    params.set_vals(2*out_dim, 3*out_dim, 0, in_dim, wfx);
    params.set_vals(2*out_dim, 3*out_dim, in_dim, in_dim + out_dim, wfh);

    params.set_vals(3*out_dim, 4*out_dim, 0, in_dim, wox);
    params.set_vals(3*out_dim, 4*out_dim, in_dim, in_dim + out_dim, woh);
  }

  void parse_params(){
    wgx = params.subMatrix(0, out_dim, 0, in_dim);
    wgh = params.subMatrix(0, out_dim, in_dim, in_dim + out_dim);
    bg = params.subMatrix(0, out_dim, in_dim + out_dim, in_dim + out_dim + 1);

    wix = params.subMatrix(out_dim, 2*out_dim, 0, in_dim);
    wih = params.subMatrix(out_dim, 2*out_dim, in_dim, in_dim + out_dim);
    bi = params.subMatrix(out_dim, 2*out_dim, in_dim + out_dim, in_dim + out_dim + 1);

    wfx = params.subMatrix(2*out_dim, 3*out_dim, 0, in_dim);
    wfh = params.subMatrix(2*out_dim, 3*out_dim, in_dim, in_dim + out_dim);
    bf = params.subMatrix(2*out_dim, 3*out_dim, in_dim + out_dim, in_dim + out_dim + 1);

    wox = params.subMatrix(3*out_dim, 4*out_dim, 0, in_dim);
    woh = params.subMatrix(3*out_dim, 4*out_dim, in_dim, in_dim + out_dim);
    bo = params.subMatrix(3*out_dim, 4*out_dim, in_dim + out_dim, in_dim + out_dim + 1);
  }

  void store_dparams(){
    d_params.set_vals(0, out_dim, 0, in_dim, d_wgx);
    d_params.set_vals(0, out_dim, in_dim, in_dim + out_dim, d_wgh);
    d_params.set_vals(0, out_dim, in_dim + out_dim, in_dim + out_dim + 1, d_bg);

    d_params.set_vals(out_dim, 2*out_dim, 0, in_dim, d_wix);
    d_params.set_vals(out_dim, 2*out_dim, in_dim, in_dim + out_dim, d_wih);
    d_params.set_vals(out_dim, 2*out_dim, in_dim + out_dim, in_dim + out_dim + 1, d_bi);

    d_params.set_vals(2*out_dim, 3*out_dim, 0, in_dim, d_wfx);
    d_params.set_vals(2*out_dim, 3*out_dim, in_dim, in_dim + out_dim, d_wfh);
    d_params.set_vals(2*out_dim, 3*out_dim, in_dim + out_dim, in_dim + out_dim + 1, d_bf);

    d_params.set_vals(3*out_dim, 4*out_dim, 0, in_dim, d_wox);
    d_params.set_vals(3*out_dim, 4*out_dim, in_dim, in_dim + out_dim, d_woh);
    d_params.set_vals(3*out_dim, 4*out_dim, in_dim + out_dim, in_dim + out_dim + 1, d_bo);
  }
};


class LinearLayer: public Layer {
public:
  LinearLayer(int, int, Layer *, OptimizerFactory *);

  void forward(const Matrix *) override;

  void backprop(const Matrix *) override;

  void release_resource() override;

private:
  int in_dim, out_dim;

  Matrix weight;
  Matrix d_weight;
  Matrix bias;
  Matrix d_bias;

  //Optimizer *weight_optim, *bias_optim;
  Optimizer *optim;

  void parse_params(){
    weight = params.subMatrix(0, out_dim, 0, in_dim);
    bias = params.subMatrix(0, out_dim, in_dim, in_dim + 1);
  }

  void store_dparams(){
    d_params.set_vals(0, out_dim, 0, in_dim, d_weight);
    d_params.set_vals(0, out_dim, in_dim, in_dim+1, d_bias);
  }
};

#endif
