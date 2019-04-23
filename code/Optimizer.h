#ifndef _OPTIMIZER_H
#define _OPTIMIZER_H

#include "Matrix.h"

class Optimizer {
public:
  virtual void update_params(Matrix &, Matrix &) = 0;
};

class OptimizerFactory {
public:
  virtual Optimizer *generate() = 0;
};

class SGD: public Optimizer {
public:
  SGD(double lr): lr(lr) {}

  void update_params(Matrix &, Matrix &) override;

private:
  double lr;
};

class SGDFactory: public OptimizerFactory {
public:
  SGDFactory(double lr): lr(lr) {}

  Optimizer *generate() override {
    return (new SGD(lr));
  }

private:
  double lr;
};

class Adam: public Optimizer {
public:
  Adam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999,
    double eps = 1e-8): lr(lr), beta1(beta1), beta2(beta2), eps(eps),
    t_beta1(1.0), t_beta2(1.0), t(0) {}

  void update_params(Matrix &, Matrix &) override;

private:
  double lr, beta1, beta2, eps;
  double t_beta1, t_beta2;
  int t;
  Matrix first_moment;
  Matrix second_moment;

  void _update_params(Matrix &, Matrix &);
};

class AdamFactory: public OptimizerFactory {
public:
  AdamFactory(double lr, double beta1, double beta2, double eps):
    lr(lr), beta1(beta1), beta2(beta2), eps(eps) {}

  Optimizer *generate() override {
    return (new Adam(lr, beta1, beta2, eps));
  }

private:
  double lr, beta1, beta2, eps;
};

#endif
