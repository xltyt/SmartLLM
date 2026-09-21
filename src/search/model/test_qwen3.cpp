/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.09.21 19:17:10
 *
\****************************************************/

#include <gtest/gtest.h>
#define C10_USE_GLOG
#include <torch/torch.h>
#include <torch/script.h>
#include <glog/logging.h>
#include <file_utils.h>
#include "utils/model_utils.h"
#include "model/qwen3.h"
#include <nlohmann/json.hpp>

Qwen3ForCausalLM *_model = NULL;
void InitModel() {
	if (_model) {
    return;
  }
  auto config = nlohmann::json::parse(std::ifstream("/data/Qwen3-0.6B/config.json"));
  _model = new Qwen3ForCausalLM(
    config["vocab_size"],
    config["hidden_size"],
    config["num_hidden_layers"],
    config["num_attention_heads"],
    config["num_key_value_heads"],
    config["head_dim"],
    config["intermediate_size"],
    config["rms_norm_eps"]
    );
  load_module_from_safetensors(*_model, "/data/Qwen3-0.6B/model.safetensors");
  _model->eval();
  LOG(INFO) << "Model Loaded";
}

TEST(Model, Qwen3) {
  auto config = nlohmann::json::parse(
  std::ifstream("/data/Qwen3-0.6B/config.json"));
  int vocab_size = config["vocab_size"];
  int hidden_size = config["hidden_size"];
  int num_layers = config["num_hidden_layers"];
  int num_heads = config["num_attention_heads"];
  int num_kv_heads = config["num_key_value_heads"];
  int head_dim = config["head_dim"];
  int intermediate_size = config["intermediate_size"];
  float rms_norm_eps = config["rms_norm_eps"];
  bool tie_word_embeddings = config.value("tie_word_embeddings", false);

  Qwen3ForCausalLM model(
    vocab_size,
    hidden_size,
    num_layers,
    num_heads,
    num_kv_heads,
    head_dim,
    intermediate_size,
    rms_norm_eps
    );
  load_module_from_safetensors(model, "/data/Qwen3-0.6B/model.safetensors");
  model.eval();
  for (const auto& p : model.named_parameters()) {
    LOG(INFO) << "Key[" << p.key() << "] Shape[" << p.value().sizes() << "]";
  }
}

TEST(Model, Qwen3ModelEmb) {
  InitModel();
  
  std::string content;
  mycommon::file_read("./test_embed.pt", content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	auto ids = data.at("ids").toTensor();
	auto ref = data.at("ref").toTensor();
	auto weight = data.at("weight").toTensor();

  auto out = _model->model->embed_tokens->forward(ids);
  double rtol = 1e-5;
	double atol = 1e-5;
	ASSERT_EQ(true, check_close("out", out, ref, rtol, atol));
}

TEST(Model, Qwen3ModelLayerInputLayer0) {
  InitModel();
  
  std::string content;
  mycommon::file_read("./test_input_layernorm_0.pt", content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	auto ids = data.at("ids").toTensor();
	auto input = data.at("input").toTensor();
	auto ref = data.at("ref").toTensor();
  
  auto out = ((Qwen3DecoderLayer *)_model->model->layers[0].get())->input_layernorm->forward(ids);
  double rtol = 1e-5;
	double atol = 1e-5;
	ASSERT_EQ(true, check_close("out", out, ref, rtol, atol));
}

TEST(Model, Qwen3ModelLayerAttn0) {
  InitModel();
  
  std::string content;
  mycommon::file_read("./test_self_attn_0.pt", content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	auto ids = data.at("ids").toTensor();
	auto hidden_states = data.at("hidden_states").toTensor();
	auto position_embeddings = std::make_tuple(data.at("position_embeddings_cos").toTensor(), data.at("position_embeddings_sin").toTensor());
	auto attention_mask = data.at("attention_mask").toTensor();
	auto ref_output_attn_output = data.at("output_attn_output").toTensor();
	auto ref_output_attn_weights = data.at("output_attn_weights").toTensor();
	
  auto [attn_output, attn_weights]  = ((Qwen3DecoderLayer *)_model->model->layers[0].get())->self_attn->forward(hidden_states, position_embeddings, attention_mask);
  double rtol = 1e-5;
	double atol = 1e-5;
	ASSERT_EQ(true, check_close("attn_output", attn_output, ref_output_attn_output, rtol, atol));
	ASSERT_EQ(true, check_close("attn_weights", attn_weights, ref_output_attn_weights, rtol, atol));
}

TEST(Model, Qwen3ModelLayer0) {
}

TEST(Model, Qwen3ModelLayerInputLayer13) {
}

TEST(Model, Qwen3ModelLayerAttn13) {
}

TEST(Model, Qwen3ModelLayer13) {
}

TEST(Model, Qwen3ModelLayerInputLayer27) {
}

TEST(Model, Qwen3ModelLayerAttn27) {
}

TEST(Model, Qwen3ModelLayer27) {
}

TEST(Model, Qwen3ModelNorm) {
}

TEST(Model, Qwen3LmHead) {
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
