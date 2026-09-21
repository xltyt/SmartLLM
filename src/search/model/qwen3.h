/****************************************************\
 *
 * Copyright (C) 2020 All Rights Reserved
 * Last modified: 2026.09.21 17:41:44
 *
\****************************************************/

#ifndef _QWEN3_H__
#define _QWEN3_H__

#define C10_USE_GLOG
#include <torch/torch.h>
#include "layers/layer_norm.h"

// Ref:
//  transformers/models/qwen3/modeling_qwen3.py
//  https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/model.safetensors

class Qwen3Attention : public torch::nn::Module {
public:
  Qwen3Attention(int hidden_size, int num_heads, int num_kv_heads, int head_dim, float rms_norm_eps);
  virtual ~Qwen3Attention();

  std::tuple<torch::Tensor, torch::Tensor> forward(
    const torch::Tensor& hidden_states,
    const std::tuple<torch::Tensor, torch::Tensor>& position_embeddings,
    const std::optional<torch::Tensor>& attention_mask
    );
  
private:  
  torch::nn::Linear q_proj{nullptr};
  torch::nn::Linear k_proj{nullptr};
  torch::nn::Linear v_proj{nullptr};
  torch::nn::Linear o_proj{nullptr};
  std::shared_ptr<RMSNorm> q_norm{nullptr};
  std::shared_ptr<RMSNorm> k_norm{nullptr};
  int _head_dim = 0;
	double _scaling = 0;
	int _num_key_value_groups = 0;
};

class Qwen3MLP : public torch::nn::Module {
public:
  Qwen3MLP(int hidden_size, int intermediate_size);
  virtual ~Qwen3MLP();

private:
  torch::nn::Linear gate_proj{nullptr};
  torch::nn::Linear up_proj{nullptr};
  torch::nn::Linear down_proj{nullptr};
};

class Qwen3DecoderLayer : public torch::nn::Module {
public:
  Qwen3DecoderLayer(int hidden_size, int num_heads, int num_kv_heads, int head_dim, int intermediate_size, float rms_norm_eps);
  virtual ~Qwen3DecoderLayer();

public:
  std::shared_ptr<RMSNorm> input_layernorm{nullptr};
  std::shared_ptr<Qwen3Attention> self_attn{nullptr};
  std::shared_ptr<RMSNorm> post_attention_layernorm{nullptr};
  std::shared_ptr<Qwen3MLP> mlp{nullptr};
};

class Qwen3Model : public torch::nn::Module {
public:
  Qwen3Model(int vocab_size, int hidden_size, int num_layers, int num_heads, int num_kv_heads, int head_dim, int intermediate_size, float rms_norm_eps);
  virtual ~Qwen3Model();

public:
  torch::nn::Embedding embed_tokens{nullptr};
  torch::nn::ModuleList layers{nullptr};
  std::shared_ptr<RMSNorm> norm{nullptr};
};

class Qwen3ForCausalLM : public torch::nn::Module {
public:
  Qwen3ForCausalLM(int vocab_size, int hidden_size, int num_layers, int num_heads, int num_kv_heads, int head_dim, int intermediate_size, float rms_norm_eps);
  virtual ~Qwen3ForCausalLM();

public:
  std::shared_ptr<Qwen3Model> model{nullptr};
  torch::nn::Linear lm_head{nullptr};
};

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
