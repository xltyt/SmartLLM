/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.09.18 14:52:48
 *
\****************************************************/

#include <gtest/gtest.h>
#define C10_USE_GLOG
#include <torch/torch.h>
#include <torch/script.h>
#include <glog/logging.h>
#include "utils/model_utils.h"
#include "model/qwen3.h"
#include <nlohmann/json.hpp>

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

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
