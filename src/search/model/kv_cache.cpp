#include "kv_cache.h"
#if USE_NPU
#include <torch_npu/csrc/aten/CustomFunctions.h>
#endif

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

std::pair<torch::Tensor, torch::Tensor> KVCache::update_static(
  int layer_idx,
  const torch::Tensor& k_states,
  const torch::Tensor& v_states,
  const DecodeContext& context
  ) {
  torch::NoGradGuard no_grad;
  TORCH_CHECK(layer_idx >= 0 && layer_idx < _num_layers, "Invalid KV layer index");
  TORCH_CHECK(context.context_len > 0 && context.context_len <= _max_seq_len,
              "Invalid static KV context length");
  const auto& cache = _k_cache[layer_idx];
  const auto& positions = context.positions;
  TORCH_CHECK(positions.defined() && positions.dim() == 1 && positions.numel() == 1 &&
              positions.scalar_type() == torch::kInt64 && positions.device() == cache.device(),
              "Static KV position must be one int64 value on the cache device");
  TORCH_CHECK(k_states.dim() == 4 && k_states.sizes() == v_states.sizes() &&
              k_states.size(0) == cache.size(0) && k_states.size(1) == cache.size(1) &&
              k_states.size(2) == 1 && k_states.size(3) == cache.size(3) &&
              k_states.device() == cache.device() && v_states.device() == cache.device() &&
              k_states.scalar_type() == cache.scalar_type() && v_states.scalar_type() == cache.scalar_type(),
              "Static KV update requires matching single-token K/V tensors");
#if USE_NPU
  // Update one complete head row per (batch, head), without transposing the cache.
  // Use the allocated capacity as the row stride; context_len can be smaller.
  // Keep positions on device so graph replay can change the destination slot.
  const int64_t rows = cache.size(0) * cache.size(1);
  auto indices = (torch::arange(rows, positions.options()) * cache.size(2) + positions).view({rows, 1});
  auto k_flat = _k_cache[layer_idx].view({-1, cache.size(3)});
  auto v_flat = _v_cache[layer_idx].view({-1, cache.size(3)});
  at_npu::native::custom_ops::npu_scatter_nd_update_(
    k_flat,
    indices,
    k_states.reshape({rows, cache.size(3)})
    );
  at_npu::native::custom_ops::npu_scatter_nd_update_(
    v_flat,
    indices,
    v_states.reshape({rows, cache.size(3)})
    );
#else
  auto indices = positions.view({1,1,1,1}).expand(k_states.sizes()).contiguous();
  _k_cache[layer_idx].scatter_(2, indices, k_states);
  _v_cache[layer_idx].scatter_(2, indices, v_states);
#endif
  return {_k_cache[layer_idx].narrow(2, 0, context.context_len),
          _v_cache[layer_idx].narrow(2, 0, context.context_len)};
}

void KVCache::reset() {
  torch::NoGradGuard no_grad;
  for (int i = 0; i < _num_layers; ++i) {
    _k_cache[i].zero_();
    _v_cache[i].zero_();
  }
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
