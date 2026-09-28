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
  
private:
  int _num_layers = 0;
  int _max_seq_len = 0;
  // All Layer pre-allocated tensor buffer: [batch, num_kv_heads, max_seq_len, head_dim]
  std::vector<torch::Tensor> _k_cache;
  std::vector<torch::Tensor> _v_cache;
};

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
