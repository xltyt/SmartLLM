/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.07.20 17:19:05
 *
\****************************************************/

#ifndef _LAYER_NORM_H__
#define _LAYER_NORM_H__

#define C10_USE_GLOG
#include <torch/torch.h>
#include <glog/logging.h>

class RMSNorm: public torch::nn::Module {
public:
  RMSNorm(int hidden_size, float eps = 1e-6);
  virtual ~RMSNorm();
	torch::Tensor forward(torch::Tensor x);
  std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor residual);
  
	void set_weight(const torch::Tensor& new_weight) {
		_weight.data() = new_weight.clone();
	}
	torch::Tensor get_weight() const {
		return _weight;
	}

private:
	torch::Tensor rms_forward(torch::Tensor x);
	std::tuple<torch::Tensor, torch::Tensor> add_rms_forward(torch::Tensor x, torch::Tensor residual);

private:
  torch::Tensor _weight;
  float _eps;
};

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
