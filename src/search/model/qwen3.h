/****************************************************\
 *
 * Copyright (C) 2020 All Rights Reserved
 * Last modified: 2026.09.22 20:09:47
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
  
public:  
  torch::Tensor forward(const torch::Tensor& x);

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
  torch::Tensor forward(
    const torch::Tensor& hidden_states,
    const std::tuple<torch::Tensor, torch::Tensor>& position_embeddings,
    const std::optional<torch::Tensor>& attention_mask
    );

public:
  std::shared_ptr<RMSNorm> input_layernorm{nullptr};
  std::shared_ptr<Qwen3Attention> self_attn{nullptr};
  std::shared_ptr<RMSNorm> post_attention_layernorm{nullptr};
  std::shared_ptr<Qwen3MLP> mlp{nullptr};
};

class Qwen3RotaryEmbedding : public torch::nn::Module {
public:
  Qwen3RotaryEmbedding(
    int64_t head_dim,
    double rope_theta = 1000000.0,
    torch::Device device = torch::kCPU);

  //
  //  Computes the inverse frequencies according to the original RoPE implementation
  //  Args:
  //      config ([`~transformers.PreTrainedConfig`]):
  //          The model configuration.
  //      device (`torch.device`):
  //          The device to use for initialization of the inverse frequencies.
  //      seq_len (`int`, *optional*):
  //          The current sequence length. Unused for this type of RoPE.
  //  Returns:
  //      Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
  //      post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
  //
  static std::tuple<torch::Tensor, double> compute_default_rope_parameters(
    int64_t head_dim,
    double rope_theta,
    torch::Device device = torch::kCPU
    );

  std::pair<torch::Tensor, torch::Tensor> forward(
    const torch::Tensor& x,
    const torch::Tensor& position_ids
    );

private:
  double _attention_scaling{1.0};
  torch::Tensor _inv_freq;
  torch::Tensor _original_inv_freq;
};


class Qwen3Model : public torch::nn::Module {
public:
  Qwen3Model(int vocab_size, int hidden_size, int num_layers, int num_heads, int num_kv_heads, int head_dim, int intermediate_size, float rms_norm_eps);
  virtual ~Qwen3Model();
  
public:
  std::tuple<torch::Tensor, torch::Tensor> prepare(
    const std::vector<int64_t>& input_ids
    );
  torch::Tensor forward(
    const std::vector<int64_t>& input_ids
    );

public:
  torch::nn::Embedding embed_tokens{nullptr};
  torch::nn::ModuleList layers{nullptr};
  std::shared_ptr<RMSNorm> norm{nullptr};
  std::shared_ptr<Qwen3RotaryEmbedding> rotary_emb{nullptr};
};

class Qwen3ForCausalLM : public torch::nn::Module {
public:
  Qwen3ForCausalLM(int vocab_size, int hidden_size, int num_layers, int num_heads, int num_kv_heads, int head_dim, int intermediate_size, float rms_norm_eps);
  virtual ~Qwen3ForCausalLM();

public:
  torch::Tensor forward(
    const std::vector<int64_t>& input_ids,
    int logits_to_keep = 1
    );

public:
  std::shared_ptr<Qwen3Model> model{nullptr};
  torch::nn::Linear lm_head{nullptr};
};

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
