#include "Optimizer.h"

void SGD::update_params(Matrix &params, Matrix &d_params){
  d_params.scale(lr);
  params = params - d_params;
}

void Adam::update_params(Matrix &params, Matrix &d_params){
  if(t == 0){
    first_moment = Matrix(params.get_row(), params.get_col());
    second_moment = Matrix(params.get_row(), params.get_col());
  }

  t_beta1 *= beta1;
  t_beta2 *= beta2;

  _update_params(params, d_params);
  ++t;
}

void Adam::_update_params(Matrix &params, Matrix &d_params) {

  first_moment = first_moment * beta1 + d_params * (1 - beta1);

  second_moment = second_moment * beta2 +
    d_params.call_func([](double x){ return x*x; }) * (1 - beta2);

  double t_lr = lr * sqrt(1.0-t_beta2) / (1.0-t_beta1);
  Matrix denom = second_moment.call_func([](double x){ return sqrt(x); });
  denom.add_scalar(eps);

  params = params - (first_moment * t_lr).entrywise_division(denom);

  // Matrix m_hat = first_moment * (1.0 / (1.0 - t_beta1));
  // Matrix v_hat = second_moment * (1.0 / (1.0 - t_beta2));
  // Matrix denom = v_hat.call_func([](double x){ return sqrt(x); });
  // denom.add_scalar(eps);
  //
  // params = params - (m_hat * lr).entrywise_division(denom);
}
