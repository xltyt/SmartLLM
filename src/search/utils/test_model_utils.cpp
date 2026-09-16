/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.09.16 16:27:58
 *
\****************************************************/

#include <gtest/gtest.h>
#define C10_USE_GLOG
#include <torch/torch.h>
#include <torch/script.h>
#include <glog/logging.h>
#include <file_utils.h>
#include "model_utils.h"

struct MyModel : torch::nn::Module {
    MyModel(std::int64_t hidden_size = 4096, std::int64_t intermediate_size = 11008) {
        fc1 = register_module("fc1", torch::nn::Linear(hidden_size, intermediate_size));
        fc2 = register_module("fc2", torch::nn::Linear(intermediate_size, hidden_size));
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_with_intermediates(
        const torch::Tensor & input) {
        auto fc1_output = fc1->forward(input);
        auto silu_output = torch::silu(fc1_output);
        auto output = fc2->forward(silu_output);
        return {fc1_output, silu_output, output};
    }

    torch::Tensor forward(const torch::Tensor & input) {
        return std::get<2>(forward_with_intermediates(input));
    }

    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
};

bool check_close(
  const std::string & name,
  const torch::Tensor & actual,
  const torch::Tensor & expected,
  double rtol,
  double atol
  ) {
  if (actual.sizes() != expected.sizes()) {
    LOG(ERROR) << name << ": shape mismatch actual=" << actual.sizes() << ", expected=" << expected.sizes();
    return false;
  }

  const auto actual_flat = actual.flatten().to(torch::kFloat64);
  const auto expected_flat = expected.flatten().to(torch::kFloat64);
  const auto difference = (actual_flat - expected_flat).abs();
  const double max_abs = difference.max().item<double>();
  const double mean_abs = difference.mean().item<double>();
  const double norm_product = actual_flat.norm().item<double>() * expected_flat.norm().item<double>();
  const double cosine = norm_product == 0.0
    ? (torch::equal(actual_flat, expected_flat) ? 1.0 : 0.0)
    : actual_flat.dot(expected_flat).item<double>() / norm_product;
  const bool close = torch::allclose(actual, expected, rtol, atol);

  LOG(INFO) << name << ": max_abs=" << max_abs << ", mean_abs=" << mean_abs << ", cosine=" << cosine << ", allclose=" << std::boolalpha << close;
  return close;
}

TEST(Utils, ModelUtils) {
  std::string content;
  mycommon::file_read("model_utils.pt", content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	int hidden_size = data.at("hidden_size").toInt();
	int intermediate_size = data.at("intermediate_size").toInt();
	auto input = data.at("input").toTensor();
	auto ref_fc1_output = data.at("fc1_output").toTensor();
	auto ref_silu_output = data.at("silu_output").toTensor();
	auto ref_output = data.at("output").toTensor();
  LOG(INFO) << "hidden_size[" << hidden_size << "] intermediate_size[" << intermediate_size << "]";
  
  MyModel model(hidden_size, intermediate_size);
  load_module_from_safetensors(model, "model.safetensors");
  model.eval();
	
  torch::InferenceMode inference_mode;
  const auto [fc1_output, silu_output, output] = model.forward_with_intermediates(input);
  
  LOG(INFO) << format_tensor(input);
  LOG(INFO) << format_tensor(fc1_output);
  LOG(INFO) << format_tensor(ref_fc1_output);
  
  double rtol = 1e-5;
	double atol = 1e-5;
	//ASSERT_EQ(true, torch::allclose(fc1_output, ref_fc1_output, rtol, atol));
	//ASSERT_EQ(true, torch::allclose(silu_output, ref_silu_output, rtol, atol));
	//ASSERT_EQ(true, torch::allclose(output, ref_output, rtol, atol));
	ASSERT_EQ(true, check_close("fc1_output", fc1_output, ref_fc1_output, rtol, atol));
	ASSERT_EQ(true, check_close("silu_output", silu_output, ref_silu_output, rtol, atol));
	ASSERT_EQ(true, check_close("output", output, ref_output, rtol, atol));
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
