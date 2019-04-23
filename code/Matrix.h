#ifndef _Matrix_H
#define _Matrix_H

#include <vector>
#include <random>
#include <iostream>

#include <cmath>

class Matrix {
public:
  Matrix():row(0), col(0) {}

  Matrix(int row, int col, double v = 0.0): row(row), col(col){
    data = std::vector<std::vector<double>>(row, std::vector<double>(col, v));
  }

  Matrix(int row, int col, double mean, double stddev): Matrix(row, col){
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, stddev);

    for(int i = 0; i < row; ++i){
      for(int j = 0; j < col; ++j){
        data[i][j] = distribution(generator);
      }
    }
  }

  void print_data() const{
    for(int i = 0; i < row; ++i){
      for(int j = 0; j < col; ++j){
        std::cout << data[i][j] << ' ';
      }
      std::cout << std::endl;
    }
  }

  void shuffle(){

  }

  void load_data(int k, int batch, int in_dim, Matrix &input, Matrix &target){
    input = Matrix(in_dim, batch);
    target = Matrix(col - in_dim, batch);

    for(int i = 0; i < in_dim; ++i){
      for(int j = 0; j < batch; ++j){
        input.data[i][j] = data[k+j][i];
      }
    }

    for(int i = 0; i < col-in_dim; ++i){
      for(int j = 0; j < batch; ++j){
        target.data[i][j] = data[k+j][in_dim + i];
      }
    }
  }

  int get_row() const{
    return row;
  }

  int get_col() const{
    return col;
  }

  double &val(int i, int j){
    return data[i][j];
  }

  double sum(double (*func)(double)){
    double result = 0;

    for(int i = 0; i < row; ++i){
      for(int j = 0; j < col; ++j){
        result += func(data[i][j]);
      }
    }

    return result;
  }

  void scale(double x){
    for(int i = 0; i < row; ++i){
      for(int j = 0; j < col; ++j){
        data[i][j] *= x;
      }
    }
  }

  void add_scalar(double x){
    for(int i = 0; i < row; ++i){
      for(int j = 0; j < col; ++j){
        data[i][j] += x;
      }
    }
  }

  // should use a more efficient algorithm
  Matrix dot(const Matrix &rhs) const{

    Matrix result(row, rhs.col);

    for(int i = 0; i < row; ++i){
      for(int j = 0; j < rhs.col; ++j){
        for(int k = 0; k < col; ++k){
          result.data[i][j] += data[i][k] * rhs.data[k][j];
        }
      }
    }

    return result;
  }


  Matrix entrywise_product(const Matrix &rhs) const{
    Matrix result(row, col);

    for(int i = 0; i < row; ++i){
      for(int j = 0; j < col; ++j){
        result.data[i][j] = data[i][j] * rhs.data[i][j];
      }
    }

    return result;
  }

  Matrix entrywise_division(const Matrix &rhs) const{
    Matrix result(row, col);

    for(int i = 0; i < row; ++i){
      for(int j = 0; j < col; ++j){
        result.data[i][j] = data[i][j] / rhs.data[i][j];
      }
    }

    return result;
  }

  Matrix transpose() const{
    Matrix result(col, row);

    for(int i = 0; i < row; ++i){
      for(int j = 0; j < col; ++j){
        result.data[j][i] = data[i][j];
      }
    }

    return result;
  }

  Matrix call_func(double (*func)(double)) const{
    Matrix result(row, col);

    for(int i = 0; i < row; ++i){
      for(int j = 0; j < col; ++j){
        result.data[i][j] = func(data[i][j]);
      }
    }

    return result;
  }

  Matrix operator-(const Matrix &rhs) const{
    Matrix result(row, col);

    for(int i = 0; i < row; ++i){
      for(int j = 0; j < col; ++j){
        result.data[i][j] = data[i][j] - rhs.data[i][j];
      }
    }

    return result;
  }

  Matrix operator+(const Matrix &rhs) const{
    Matrix result(row, col);

    for(int i = 0; i < row; ++i){
      for(int j = 0; j < col; ++j){
        result.data[i][j] = data[i][j] + rhs.data[i][j];
      }
    }

    return result;
  }

  Matrix operator*(double x) const{
    Matrix result(row, col);

    for(int i = 0; i < row; ++i){
      for(int j = 0; j < col; ++j){
        result.data[i][j] = data[i][j] * x;
      }
    }

    return result;
  }

  Matrix get_the_col(int k) const{
    Matrix result(row, 1);

    for(int i = 0; i < row; ++i) result.data[i][0] = data[i][k];

    return result;
  }

  Matrix get_the_row(int k) const{
    Matrix result(1, col);

    for(int i = 0; i < col; ++i) result.data[0][i] = data[k][i];

    return result;
  }

  void push_back_row(const Matrix &in){
    data.push_back(in.data[0]);
    if(row == 0) col = in.get_col();
    ++row;
  }

  void insert_to_col(int k, const Matrix &in){
    for(int i = 0; i < row; ++i){
      data[i][k] = in.data[i][0];
    }
  }

  Matrix get_range_rows(int s, int e){
    Matrix result(e-s, col);

    for(int i = s; i < e; ++i){
      for(int j = 0; j < col; ++j){
        result.data[i-s][j] = data[i][j];
      }
    }

    return result;
  }

  Matrix subMatrix(int s_row, int e_row, int s_col, int e_col) const{
    Matrix result(e_row-s_row, e_col-s_col);

    for(int i = s_row; i < e_row; ++i){
      for(int j = s_col; j < e_col; ++j){
        result.data[i-s_row][j-s_col] = data[i][j];
      }
    }

    return result;
  }

  void set_vals(int s_row, int e_row, int s_col, int e_col, const Matrix &in){
    for(int i = s_row; i < e_row; ++i){
      for(int j = s_col; j < e_col; ++j){
        data[i][j] = in.data[i-s_row][j-s_col];
      }
    }
  }

private:
  int row, col;
  std::vector<std::vector<double>> data;
};

#endif
