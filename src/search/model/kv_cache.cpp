#include "kv_cache.h"

KVCache::KVCache(
  int num_layers,
  int batch_size,
  int num_kv_heads,
  int head_dim,
  int max_seq_len,
  torch::Dtype dtype /*= torch::kFloat32*/,
  torch::Device device /*= torch::kCPU*/
  ) {
  _num_layers = num_layers;
  _max_seq_len = max_seq_len;

  auto options = torch::TensorOptions().dtype(dtype).device(device);
  _k_cache.resize(num_layers);
  _v_cache.resize(num_layers);

  for (int i = 0; i < num_layers; ++i) {
    _k_cache[i] = torch::zeros({batch_size, num_kv_heads, max_seq_len, head_dim}, options);
    _v_cache[i] = torch::zeros({batch_size, num_kv_heads, max_seq_len, head_dim}, options);
  }
}

KVCache::~KVCache() {
}
  
std::pair<torch::Tensor, torch::Tensor> KVCache::update(
  int layer_idx,
  const torch::Tensor& k_states,
  const torch::Tensor& v_states,
  int64_t start_pos
) {
  torch::NoGradGuard no_grad;
  // [batch, num_kv_heads, seq_len, head_dim]
  int64_t seq_len = k_states.size(2);
  
  if (layer_idx < 0 || layer_idx >= _num_layers) {
    throw std::out_of_range("KVCache: Invalid layer_idx: " + std::to_string(layer_idx));
  }
  if (start_pos + seq_len > _max_seq_len) {
    throw std::length_error("KVCache: overflow! max_seq_len: " + std::to_string(_max_seq_len) + ", required: " + std::to_string(start_pos + seq_len));
  }

  // Zero-Allocation Copy
  _k_cache[layer_idx].slice(
    2,                  // Dim
    start_pos,          // Start
    start_pos + seq_len // End
    ).copy_(k_states);
  _v_cache[layer_idx].slice(
    2,                  // Dim
    start_pos,          // Start
    start_pos + seq_len // End
    ).copy_(v_states);

  // All Valid Tensor Util Current Position
  auto valid_k = _k_cache[layer_idx].slice(
    2,
    0,
    start_pos + seq_len
    );
  auto valid_v = _v_cache[layer_idx].slice(
    2,
    0,
    start_pos + seq_len
    );

  return {valid_k, valid_v};

}

void KVCache::reset() {
  torch::NoGradGuard no_grad;
  for (int i = 0; i < _num_layers; ++i) {
    _k_cache[i].zero_();
    _v_cache[i].zero_();
  }
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
