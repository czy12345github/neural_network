#ifndef _LOSSFUNCTION_H
#define _LOSSFUNCTION_H

#include "Matrix.h"

class LossFunction {
public:
  virtual double compute_loss(const Matrix &out, const Matrix &target) = 0;

  virtual double compute_loss(const Matrix &out, const Matrix &target, Matrix &deriv) = 0;
};

class MSELoss: public LossFunction {
public:
  double compute_loss(const Matrix &out, const Matrix &target) override;

  double compute_loss(const Matrix &out, const Matrix &target, Matrix &deriv) override;
};

#endif
