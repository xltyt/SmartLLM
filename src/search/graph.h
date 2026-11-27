#ifndef SMARTLLM_GRAPH_H
#define SMARTLLM_GRAPH_H

#define C10_USE_GLOG
#include <torch/torch.h>

#if USE_NPU
#include <memory>
#include <optional>
#include <torch_npu/csrc/core/npu/NPUGraph.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

class Qwen3ForCausalLM;
class KVCache;

// One fixed context length, batch=1, serialized calls. Prefill remains eager.
// Prepare weights/RoPE before first forward. Model/cache must outlive this object;
// do not reload/move weights or reallocate cache until this object is destroyed.
// Returned logits are independent of later replays. Same-stream cache reset is safe.
class DecodeGraph {
public:
  static bool supported();
  DecodeGraph(Qwen3ForCausalLM& model, KVCache& cache, int64_t context_len);
  ~DecodeGraph();
  DecodeGraph(const DecodeGraph&) = delete;
  DecodeGraph& operator=(const DecodeGraph&) = delete;

  torch::Tensor forward(int64_t token_id, int64_t start_pos);
  int64_t capture_count() const;
  int64_t replay_count() const;

private:
  Qwen3ForCausalLM& _model;
  torch::Device _device;
  int64_t _context_len;
  int64_t _captures = 0;
  int64_t _replays = 0;
  bool _capture_failed = false;

#if USE_NPU
  void update_inputs(int64_t token_id, int64_t position);
  void order_stream(c10_npu::NPUStream current);
  void capture(int64_t token_id, int64_t position, c10_npu::NPUStream caller_stream);

  torch::Tensor _host_input;
  torch::Tensor _device_input;
  torch::Tensor _token;
  torch::Tensor _position;
  torch::Tensor _output;
  std::optional<c10_npu::NPUStream> _previous_stream;
  std::optional<c10_npu::NPUStream> _capture_stream;
  // Allocate lazily under the device guard; destroy before its IO references.
  std::unique_ptr<c10_npu::NPUGraph> _graph;
#endif
};

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
