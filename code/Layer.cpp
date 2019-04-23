#include "Layer.h"

#include <iostream>

#include <cstdlib>
#include <cmath>

using namespace std;

void Layer::forward(const Matrix *input){
  if(input != NULL) in = input;
  else in = &(pre_layer->out);
}

void Layer::backprop(const Matrix *d_output){
  if(d_output != NULL) d_out = d_output;
  else d_out = &(next_layer->d_in);
}

LinearLayer::LinearLayer(int in_dim, int out_dim, Layer *pre,
  OptimizerFactory *optimFactory): Layer(pre, out_dim, in_dim+1), in_dim(in_dim), out_dim(out_dim),
  weight(out_dim, in_dim, 0.0, sqrt(2.0 / in_dim)) {

  params.set_vals(0, out_dim, 0, in_dim, weight);
  optim = optimFactory->generate();
}

void LinearLayer::forward(const Matrix *input){
  Layer::forward(input);

  parse_params();

  out = weight.dot(*in) + bias.dot(Matrix(1, in->get_col(), 1.0));

  if(act != NULL)
    out = act->activate(out);
}


void LinearLayer::backprop(const Matrix *d_output){
  Layer::backprop(d_output);

  Matrix d_linear_out;
  if(act != NULL) d_linear_out = d_out->entrywise_product(act->derivate(out));
  else d_linear_out = *d_out;

  d_bias = d_linear_out.dot(Matrix(d_linear_out.get_col(), 1, 1.0));
  d_weight = d_linear_out.dot(in->transpose());

  if(!is_first_layer){
    d_in = weight.transpose().dot(d_linear_out);
  }

  store_dparams();

  optim->update_params(params, d_params);
}


void LinearLayer::release_resource(){
  delete optim;
  if(act != NULL) delete act;
}


LSTMLayer::LSTMLayer(int in_dim, int out_dim, Layer *pre,
OptimizerFactory *optimFactory): Layer(pre, 4*out_dim, out_dim+in_dim+1),
in_dim(in_dim), out_dim(out_dim),
wgx(out_dim, in_dim), wgh(out_dim, out_dim), bg(out_dim, 1),
wix(out_dim, in_dim), wih(out_dim, out_dim), bi(out_dim, 1),
wfx(out_dim, in_dim), wfh(out_dim, out_dim), bf(out_dim, 1),
wox(out_dim, in_dim), woh(out_dim, out_dim), bo(out_dim, 1)
{
  reset_params(wgx);
  reset_params(wgh);

  reset_params(wix);
  reset_params(wih);

  reset_params(wfx);
  reset_params(wfh);

  reset_params(wox);
  reset_params(woh);

  init_params();

  act_tanh = ActivatorFactory::tanhActivator();
  act_sigmoid = ActivatorFactory::sigmoidActivator();

  optim = optimFactory->generate();
}

void LSTMLayer::reset_params(Matrix &params){
  double stdv = 1.0 / sqrt(wgx.get_row());

  std::random_device rd;
  std::uniform_real_distribution<double> dis(-stdv, stdv);

  for(int i = 0; i < params.get_row(); ++i){
    for(int j = 0; j < params.get_col(); ++j){
      params.val(i, j) = dis(rd);
    }
  }
}


void LSTMLayer::forward(const Matrix *input){
  Layer::forward(input);

  clear_matrix();

  parse_params();

  Matrix gt, it, ft, ot;

  Matrix x, pre_h, pre_s;

  Matrix st, ht;

  for(int i = 0; i < in->get_col(); ++i){
    x = in->get_the_col(i);
    pre_h = h.get_the_row(i).transpose();
    pre_s = state.get_the_row(i).transpose();

    gt = wgx.dot(x) + wgh.dot(pre_h) + bg;
    gt = act_tanh->activate(gt);
    seq_g.push_back_row(gt.transpose());

    it = wix.dot(x) + wih.dot(pre_h) + bi;
    it = act_sigmoid->activate(it);
    seq_i.push_back_row(gt.transpose());

    ft = wfx.dot(x) + wfh.dot(pre_h) + bf;
    ft = act_sigmoid->activate(ft);
    seq_f.push_back_row(ft.transpose());

    ot = wox.dot(x) + woh.dot(pre_h) + bo;
    ot = act_sigmoid->activate(ot);
    seq_o.push_back_row(ot.transpose());

    st = gt.entrywise_product(it) + pre_s.entrywise_product(ft);
    ht = st.entrywise_product(ot);

    state.push_back_row(st.transpose());
    h.push_back_row(ht.transpose());
  }

  out = (h.get_range_rows(1, h.get_row())).transpose();
  //std::cout << "out row: " << out.get_row() << ", col: " << out.get_col() << std::endl;
}


void LSTMLayer::backprop(const Matrix *d_output){
  Layer::backprop(d_output);

  Matrix diff_h(out.get_row(), 1);
  Matrix diff_s(out.get_row(), 1);

  Matrix gt, it, ft, ot;

  Matrix xt, ht;

  Matrix st, pre_s;

  Matrix d_s, d_o, d_i, d_g, d_f;
  Matrix d_g_tanh, d_i_sig, d_f_sig, d_o_sig;

  int x_row = wgx.get_row(), x_col = wgx.get_col();
  int h_row = wgh.get_row(), h_col = wgh.get_col();
  d_wgx = Matrix(x_row, x_col);
  d_wgh = Matrix(h_row, h_col);
  d_wix = Matrix(x_row, x_col);
  d_wih = Matrix(h_row, h_col);
  d_wfx = Matrix(x_row, x_col);
  d_wfh = Matrix(h_row, h_col);
  d_wox = Matrix(x_row, x_col);
  d_woh = Matrix(h_row, h_col);
  d_bg = Matrix(x_row, 1);
  d_bi = Matrix(x_row, 1);
  d_bf = Matrix(x_row, 1);
  d_bo = Matrix(x_row, 1);

  if(!is_first_layer){
    d_in = Matrix(in->get_row(), in->get_col());
  }

  for(int i = d_out->get_col() - 1; i >= 0; --i){
    diff_h = diff_h + d_out->get_the_col(i);

    gt = seq_g.get_the_row(i).transpose();
    it = seq_i.get_the_row(i).transpose();
    ft = seq_f.get_the_row(i).transpose();
    ot = seq_o.get_the_row(i).transpose();
    //std::cout << "restore gt, it, ft and ot successfully" << std::endl;

    st = state.get_the_row(i+1).transpose();
    pre_s = state.get_the_row(i).transpose();
    //std::cout << "restore st and pre_s successfully" << std::endl;

    d_s = ot.entrywise_product(diff_h) + diff_s;
    d_o = st.entrywise_product(diff_h);
    d_i = gt.entrywise_product(d_s);
    d_g = it.entrywise_product(d_s);
    d_f = pre_s.entrywise_product(d_s);

    d_g_tanh = d_g.entrywise_product(act_tanh->derivate(gt));
    d_i_sig = d_i.entrywise_product(act_sigmoid->derivate(it));
    d_f_sig = d_f.entrywise_product(act_sigmoid->derivate(ft));
    d_o_sig = d_o.entrywise_product(act_sigmoid->derivate(ot));
    //std::cout << "compute d_ successfully" << std::endl;

    // input of this time step
    xt = in->get_the_col(i).transpose();
    ht = h.get_the_row(i);

    d_wgx = d_wgx + d_g_tanh.dot(xt);
    d_wgh = d_wgh + d_g_tanh.dot(ht);
    d_wix = d_wix + d_i_sig.dot(xt);
    d_wih = d_wih + d_i_sig.dot(ht);
    d_wfx = d_wfx + d_f_sig.dot(xt);
    d_wfh = d_wfh + d_f_sig.dot(ht);
    d_wox = d_wox + d_o_sig.dot(xt);
    d_woh = d_woh + d_o_sig.dot(ht);
    d_bg = d_bg + d_g_tanh;
    d_bi = d_bi + d_i_sig;
    d_bf = d_bf + d_f_sig;
    d_bo = d_bo + d_o_sig;
    //std::cout << "update deriv successfully" << std::endl;

    diff_s = d_s.entrywise_product(ft);
    diff_h = wgh.transpose().dot(d_g_tanh);
    diff_h = diff_h + wih.transpose().dot(d_i_sig);
    diff_h = diff_h + wfh.transpose().dot(d_f_sig);
    diff_h = diff_h + woh.transpose().dot(d_o_sig);
    //std::cout << "diff_h and diff_s" << std::endl;

    if(!is_first_layer){
      Matrix d_x = wgx.transpose().dot(d_g_tanh);
      d_x = d_x + wix.transpose().dot(d_i_sig);
      d_x = d_x + wfx.transpose().dot(d_f_sig);
      d_x = d_x + wox.transpose().dot(d_o_sig);
      //std::cout << "d_x" << std::endl;
      d_in.insert_to_col(i, d_x);
    }
  }

  store_dparams();

  optim->update_params(params, d_params);
}

void LSTMLayer::release_resource(){
  delete optim;
}
