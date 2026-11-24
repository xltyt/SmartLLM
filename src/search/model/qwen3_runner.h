#ifndef _QWEN3_RUNNER_H
#define _QWEN3_RUNNER_H

#include <memory>
#include <string>
#include <vector>
#include "qwen3.h"
#include "qwen3_graph.h"

// Batch=1, serialized calls on one execution stream. The caller initializes
// the device runtime before construction and finalizes it after destruction.
// Sampling and the generation loop remain with the caller.
class Qwen3Runner {
public:
  Qwen3Runner(
    const std::string& model_path,
    torch::Device device,
    torch::Dtype dtype,
    int max_seq_len = 2048,
    bool decode_graph = false
    );
  Qwen3Runner(const Qwen3Runner&) = delete;
  Qwen3Runner& operator=(const Qwen3Runner&) = delete;

  // Start a prompt at position zero, returning only the last token's logits.
  torch::Tensor prefill(const std::vector<int64_t>& input_ids);
  torch::Tensor decode(int64_t token_id, int64_t position);
  // Call between requests. Keeps weights, buffers and any captured graph.
  void reset();

  int64_t capacity() const { return _kv_cache->capacity(); }
  bool graph_enabled() const { return _decode_graph != nullptr; }
  int64_t capture_count() const;
  int64_t replay_count() const;

private:
  int64_t _vocab_size;
  // Destruction order: graph first, then model, then KV cache.
  std::unique_ptr<KVCache> _kv_cache;
  std::unique_ptr<Qwen3ForCausalLM> _model;
  torch::Tensor _decode_token;
  torch::Tensor _decode_position;
  std::unique_ptr<Qwen3DecodeGraph> _decode_graph;
};

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
