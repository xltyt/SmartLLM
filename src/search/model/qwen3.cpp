#include "qwen3.h"
#include "safetensors.hh"
#include <fstream>
#include <vector>

// ============ Attention ============
Qwen3Attention::Qwen3Attention(int hidden_size, int num_heads, int num_kv_heads, int head_dim) {
  //int head_dim = hidden_size / num_heads;
  //LOG(INFO) << "Qwen3Attention hidden_size[" << hidden_size << "] num_heads[" << num_heads << "] head_dim[" << head_dim << "]";
  this->q_proj = register_module("q_proj", torch::nn::Linear(
    torch::nn::LinearOptions(hidden_size, num_heads * head_dim).bias(false)
    ));
  this->k_proj = register_module("k_proj", torch::nn::Linear(
    torch::nn::LinearOptions(hidden_size, num_kv_heads * head_dim).bias(false)
    ));
  this->v_proj = register_module("v_proj", torch::nn::Linear(
    torch::nn::LinearOptions(hidden_size, num_kv_heads * head_dim).bias(false)
    ));
  this->o_proj = register_module("o_proj", torch::nn::Linear(
    torch::nn::LinearOptions(num_heads * head_dim, hidden_size).bias(false)
    ));
}

Qwen3Attention::~Qwen3Attention() {
}
  
// ============ MLP ============
Qwen3MLP::Qwen3MLP(int hidden_size, int intermediate_size) {
  this->gate_proj = register_module("gate_proj", torch::nn::Linear(
    torch::nn::LinearOptions(hidden_size, intermediate_size).bias(false)
    ));
  this->up_proj = register_module("up_proj", torch::nn::Linear(
    torch::nn::LinearOptions(hidden_size, intermediate_size).bias(false)
    ));
  this->down_proj = register_module("down_proj", torch::nn::Linear(
    torch::nn::LinearOptions(intermediate_size, hidden_size).bias(false)
    ));
}

Qwen3MLP::~Qwen3MLP() {
}

// ============ Decoder Layer ============
Qwen3DecoderLayer::Qwen3DecoderLayer(int hidden_size, int num_heads, int num_kv_heads, int head_dim, int intermediate_size, float rms_norm_eps) {
  this->self_attn = register_module("self_attn", std::make_shared<Qwen3Attention>(hidden_size, num_heads, num_kv_heads, head_dim));
  this->mlp = register_module("mlp", std::make_shared<Qwen3MLP>(hidden_size, intermediate_size));
  this->input_layernorm = register_module("input_layernorm", std::make_shared<RMSNorm>(hidden_size, rms_norm_eps));
  this->post_attention_layernorm = register_module("post_attention_layernorm", std::make_shared<RMSNorm>(hidden_size, rms_norm_eps));
}

Qwen3DecoderLayer::~Qwen3DecoderLayer() {
}

// ============ Model ============
Qwen3Model::Qwen3Model(int vocab_size, int hidden_size, int num_layers, int num_heads, int num_kv_heads, int head_dim, int intermediate_size, float rms_norm_eps) {
  this->embed_tokens = register_module("embed_tokens", torch::nn::Embedding(vocab_size, hidden_size));

  this->layers = register_module("layers", torch::nn::ModuleList());
  for (int i = 0; i < num_layers; i++) {
    layers->push_back(std::make_shared<Qwen3DecoderLayer>(
      hidden_size,
      num_heads,
      num_kv_heads,
      head_dim,
      intermediate_size,
      rms_norm_eps
      ));
  }

  this->norm = register_module("norm", std::make_shared<RMSNorm>(hidden_size, rms_norm_eps));
}

Qwen3Model::~Qwen3Model() {
}
  
// ============ Top-level Model ============
Qwen3ForCausalLM::Qwen3ForCausalLM::Qwen3ForCausalLM(int vocab_size, int hidden_size, int num_layers, int num_heads, int num_kv_heads, int head_dim, int intermediate_size, float rms_norm_eps) {
  this->model = register_module("model", std::make_shared<Qwen3Model>(
    vocab_size,
    hidden_size,
    num_layers,
    num_heads,
    num_kv_heads,
    head_dim,
    intermediate_size,
    rms_norm_eps
    ));

  this->lm_head = register_module("lm_head", torch::nn::Linear(
    torch::nn::LinearOptions(hidden_size, vocab_size).bias(false)
    ));
}

Qwen3ForCausalLM::~Qwen3ForCausalLM() {
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
