/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.07.21 18:27:53
 *
\****************************************************/

#include <gtest/gtest.h>
#define C10_USE_GLOG
#include <torch/torch.h>
#include <torch/script.h>
#include <glog/logging.h>
#include <file_utils.h>
#include "activation.h"
#include "layer_norm.h"
#include "sampler.h"

TEST(Layer, Activation) {
	std::string file = "activation.pt";
  
  std::string content;
  mycommon::file_read(file, content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	auto input = data.at("input").toTensor();
	auto expected = data.at("output").toTensor();
  SiluAndMul layer;
	auto result = layer.forward(input);
	//LOG(INFO) << "Expected[" << expected << "] Result[" << result << "]";
	double rtol = 1e-5;
	double atol = 1e-5;
	bool ok = torch::allclose(result, expected, rtol, atol);
	ASSERT_EQ(ok, true);
	//auto diff = (result - expected).abs();
	//LOG(INFO) << "Diff[" << diff.max().item<double>() << "]";
}

TEST(Layer, RMSNorm) {
	std::string file = "layer_norm.pt";
  
  std::string content;
  mycommon::file_read(file, content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	auto input = data.at("input").toTensor();
	auto weight = data.at("weight").toTensor();
	auto residual = data.at("residual").toTensor();
	auto output = data.at("output").toTensor();
	auto output2 = data.at("output2").toTensor();
	auto residual2 = data.at("residual2").toTensor();
	auto input_zero = data.at("input_zero").toTensor();
	auto output_zero = data.at("output_zero").toTensor();
	auto input_large = data.at("input_large").toTensor();
	auto output_large = data.at("output_large").toTensor();
	auto input_fp16 = data.at("input_fp16").toTensor();
	auto output_fp16 = data.at("output_fp16").toTensor();
  
  int hidden_size = 1024;
  int batch_size = 2;
  int seq_len = 10;
  float eps = 1e-6;
	double rtol = 1e-5;
	double atol = 1e-5;
  RMSNorm rms_layer(hidden_size, eps);
	rms_layer.set_weight(weight);
	ASSERT_EQ(true, torch::allclose(rms_layer.forward(input.clone()), output, rtol, atol));
  std::tuple<torch::Tensor, torch::Tensor> output_residual_result = rms_layer.forward(input.clone(), residual.clone());
	ASSERT_EQ(true, torch::allclose(std::get<0>(output_residual_result), output2, rtol, atol));
	ASSERT_EQ(true, torch::allclose(std::get<1>(output_residual_result), residual2, rtol, atol));
	ASSERT_EQ(true, torch::allclose(rms_layer.forward(input_zero.clone()), output_zero, rtol, atol));
	ASSERT_EQ(true, torch::allclose(rms_layer.forward(input_large.clone()), output_large, rtol, atol));
}

TEST(Layer, RotaryEmbedding) {
	std::string file = "rotary_embedding.pt";
}

TEST(Layer, Sampler) {
	std::string file = "sampler.pt";
  
  std::string content;
  mycommon::file_read(file, content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	auto logits = data.at("logits").toTensor();
	auto temperatures = data.at("temperatures").toTensor();
	auto output = data.at("output").toTensor();
	double rtol = 1e-5;
	double atol = 1e-5;
  Sampler sampler;
	auto result = sampler.forward(logits, temperatures);
	ASSERT_EQ(true, torch::allclose(result, output, rtol, atol));
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
