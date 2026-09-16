/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.07.21 15:30:29
 *
\****************************************************/

#include "layer_norm.h"
  
RMSNorm::RMSNorm(int hidden_size, float eps /*= 1e-6*/) {
	_weight = register_parameter("weight", torch::ones({hidden_size}));
	_eps = eps;
}

RMSNorm::~RMSNorm() {
}

torch::Tensor RMSNorm::forward(torch::Tensor x) {
	return rms_forward(x);
}

std::tuple<torch::Tensor, torch::Tensor> RMSNorm::forward(torch::Tensor x, torch::Tensor residual) {
	return add_rms_forward(x, residual);
}

torch::Tensor RMSNorm::rms_forward(torch::Tensor x) {
	auto orig_dtype = x.dtype();
	x = x.to(torch::kFloat32);
	auto var = x.pow(2).mean(-1, true);
	x.mul_(var.add(_eps).rsqrt());
	x = x.to(orig_dtype).mul_(_weight);
	return x;
}

std::tuple<torch::Tensor, torch::Tensor> RMSNorm::add_rms_forward(torch::Tensor x, torch::Tensor residual) {
	auto orig_dtype = x.dtype();
	x = x.to(torch::kFloat32).add_(residual.to(torch::kFloat32));
	auto new_residual = x.to(orig_dtype);
	auto var = x.pow(2).mean(-1, true);
	x.mul_(var.add(_eps).rsqrt());
	x = x.to(orig_dtype).mul_(_weight);
	return std::make_tuple(x, new_residual);
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
