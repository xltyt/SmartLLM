/****************************************************\
 *
 * Copyright (C) 2020 All Rights Reserved
 * Last modified: 2026.09.28 20:05:09
 *
\****************************************************/

#ifndef _KV_CACHE_H__
#define _KV_CACHE_H__

#define C10_USE_GLOG
#include <torch/torch.h>

// Host metadata is fixed during capture; position values stay on the device.
struct DecodeContext {
  torch::Tensor positions;
  int64_t context_len;
};

class KVCache {
public:
  KVCache(
    int num_layers,
    int batch_size,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    torch::Dtype dtype = torch::kFloat32,
    torch::Device device = torch::kCPU
    );
  virtual ~KVCache();
  
public:
  // Write Layer Index's KV, Return All KV Before Current Layer(Contain)
  // k, v shape: [batch_size, num_kv_heads, seq_len, head_dim]
  // Ret pair<Tensor, Tensor> Shape: [batch_size, num_kv_heads, start_pos + seq_len, head_dim]
  std::pair<torch::Tensor, torch::Tensor> update(
    int layer_idx,
    const torch::Tensor& k_states,
    const torch::Tensor& v_states,
    int64_t start_pos
  );

  // Reset Cache For Next Request
  void reset();

  int64_t capacity() const { return _max_seq_len; }
  int64_t batch_size() const { return _k_cache.front().size(0); }

  // Single-token decode with a fixed-size read view. Keep context_len fixed
  // across replays and update the contents of the same device position tensor.
  // The caller validates 0 <= position < context_len before capture/replay;
  // do not read the position value back to the host here.
  std::pair<torch::Tensor, torch::Tensor> update_static(
    int layer_idx,
    const torch::Tensor& k_states,
    const torch::Tensor& v_states,
    const DecodeContext& context);

  
private:
  int _num_layers = 0;
  int _max_seq_len = 0;
  // All Layer pre-allocated tensor buffer: [batch, num_kv_heads, max_seq_len, head_dim]
  std::vector<torch::Tensor> _k_cache;
  std::vector<torch::Tensor> _v_cache;
};

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
