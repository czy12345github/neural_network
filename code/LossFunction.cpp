#include "LossFunction.h"

#include <iostream>

using namespace std;

double MSELoss::compute_loss(const Matrix &out, const Matrix &target){
  return (out - target).sum([](double x){ return x*x; });
}

double MSELoss::compute_loss(const Matrix &out, const Matrix &target, Matrix &deriv){
  double loss;

  deriv = out - target;
  loss = deriv.sum([](double x){ return x*x; });

  deriv.scale(2.0);
  return loss;
}
