/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.09.18 18:08:50
 *
\****************************************************/

#include <gtest/gtest.h>
#define C10_USE_GLOG
#include <torch/torch.h>
#include <torch/script.h>
#include <glog/logging.h>
#include <safetensors.hh>
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

TEST(Utils, SafeTensors) {
  std::string file_path = "model.safetensors";
  // Parse
  std::string warn;
  std::string err;
  safetensors::safetensors_t st;
  if (!safetensors::load_from_file(file_path, &st, &warn, &err)) {
    throw std::runtime_error("Failed to parse safetensors[" + err + "]");
  }
  if (!safetensors::validate_data_offsets(st, err)) {
    throw std::runtime_error("Invalid data_offsets[" + err + "]");
  }

  const std::vector<std::string> tensor_keys = st.tensors.keys();
  for (auto &key : tensor_keys) {
    LOG(INFO) << "Key[" << key << "]";
  }
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
  //model.to(torch::kCUDA);
	
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
