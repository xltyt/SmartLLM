/****************************************************\
 *
 * Copyright (C) 2020 All Rights Reserved
 * Last modified: 2026.09.18 14:31:39
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
  Qwen3Attention(int hidden_size, int num_heads, int num_kv_heads, int head_dim);
  virtual ~Qwen3Attention();
  
private:  
  torch::nn::Linear q_proj{nullptr};
  torch::nn::Linear k_proj{nullptr};
  torch::nn::Linear v_proj{nullptr};
  torch::nn::Linear o_proj{nullptr};
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

private:
  std::shared_ptr<Qwen3Attention> self_attn{nullptr};
  std::shared_ptr<Qwen3MLP> mlp{nullptr};
  std::shared_ptr<RMSNorm> input_layernorm{nullptr};
  std::shared_ptr<RMSNorm> post_attention_layernorm{nullptr};
};

class Qwen3Model : public torch::nn::Module {
public:
  Qwen3Model(int vocab_size, int hidden_size, int num_layers, int num_heads, int num_kv_heads, int head_dim, int intermediate_size, float rms_norm_eps);
  virtual ~Qwen3Model();

private:
  torch::nn::Embedding embed_tokens{nullptr};
  torch::nn::ModuleList layers{nullptr};
  std::shared_ptr<RMSNorm> norm{nullptr};
};

class Qwen3ForCausalLM : public torch::nn::Module {
public:
  Qwen3ForCausalLM(int vocab_size, int hidden_size, int num_layers, int num_heads, int num_kv_heads, int head_dim, int intermediate_size, float rms_norm_eps);
  virtual ~Qwen3ForCausalLM();

private:
  std::shared_ptr<Qwen3Model> model{nullptr};
  torch::nn::Linear lm_head{nullptr};
};

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
