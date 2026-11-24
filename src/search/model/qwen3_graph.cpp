#include "qwen3_graph.h"
#include "qwen3.h"

#if USE_NPU
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#endif

bool Qwen3DecodeGraph::supported() {
#if USE_NPU
  return true;
#else
  return false;
#endif
}

Qwen3DecodeGraph::Qwen3DecodeGraph(Qwen3ForCausalLM& model, KVCache& cache, int64_t context_len)
  : _model(model), _device(model.model->embed_tokens->weight.device()),
    _context_len(context_len) {
  TORCH_CHECK(supported(), "Decode graph requires an NPU build");
  TORCH_CHECK(_device.is_privateuseone(), "Decode graph requires an NPU model");
  TORCH_CHECK(cache.batch_size() == 1 && context_len > 0 && context_len <= cache.capacity(),
              "Decode graph requires batch=1 and a valid fixed context length");
}

Qwen3DecodeGraph::~Qwen3DecodeGraph() {
#if USE_NPU
  try {
    if (_previous_stream) _previous_stream->synchronize();
    if (_capture_stream) _capture_stream->synchronize();
  }
  catch (const std::exception& error) {
    TORCH_WARN("Decode graph teardown synchronization failed: ", error.what());
  }
#endif
}

#if USE_NPU
void Qwen3DecodeGraph::update_inputs(int64_t token_id, int64_t position) {
  auto* values = _host_input.data_ptr<int64_t>();
  values[0] = token_id;
  values[1] = position;
  // Deliberately outside capture. Blocking copy preserves host buffer life.
  _device_input.copy_(_host_input, /*non_blocking=*/false);
}

void Qwen3DecodeGraph::order_stream(c10_npu::NPUStream current) {
  // Static IO and the KV cache cannot be used concurrently on two streams.
  if (_previous_stream && *_previous_stream != current) _previous_stream->synchronize();
  _previous_stream = current;
}

void Qwen3DecodeGraph::capture(int64_t token_id, int64_t position,
                              c10_npu::NPUStream caller_stream) {
  _capture_stream = c10_npu::getNPUStreamFromPool(_device.index());
  _graph = std::make_unique<c10_npu::NPUGraph>();
  _host_input = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  _device_input = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64).device(_device));
  _token = _device_input.narrow(0, 0, 1).view({1, 1});
  _position = _device_input.narrow(0, 1, 1);
  update_inputs(token_id, position);
  caller_stream.synchronize();  // Includes the preceding eager prefill.
  c10_npu::NPUStreamGuard stream_guard(*_capture_stream);
  DecodeContext context{_position, _context_len};
  // Repeatedly overwrite the same current KV slot, never advance the cache.
  // Lazy initialization/compilation must finish before capture begins.
  for (int i = 0; i < 3; ++i) {
    (void)_model.forward_static_decode(_token, context);
  }
  _capture_stream->synchronize();
  bool begun = false;
  bool ending = false;
  try {
    _graph->capture_begin();
    begun = true;
    _output = _model.forward_static_decode(_token, context);
    ending = true;
    _graph->capture_end();
    begun = false;
  }
  catch (...) {
    _capture_failed = true;
    if (begun && !ending) {
      try {
        _graph->capture_end();
      }
      catch (...) {
        TORCH_WARN("Failed to close NPU capture; runtime recovery is required");
      }
    }
    // Do not silently execute eager work on a potentially invalid capture stream.
    throw;
  }
  _capture_stream->synchronize();
  ++_captures;
}
#endif

torch::Tensor Qwen3DecodeGraph::forward(int64_t token_id, int64_t start_pos) {
  torch::NoGradGuard no_grad;
  TORCH_CHECK(!_capture_failed, "Previous NPU graph capture failed; recover the runtime before continuing");
  TORCH_CHECK(!_model.is_training(), "Decode graph requires eval mode");
  TORCH_CHECK(start_pos >= 0 && start_pos < _context_len, "Decode position exceeds graph context");
  TORCH_CHECK(token_id >= 0 && token_id < _model.model->embed_tokens->weight.size(0),
              "Decode token is outside the vocabulary");
#if USE_NPU
  c10_npu::NPUGuard device_guard(_device);
  auto stream = c10_npu::getCurrentNPUStream(_device.index());
  order_stream(stream);
  try {
    if (_captures == 0) {
      capture(token_id, start_pos, stream);
    }
    else {
      update_inputs(token_id, start_pos);
    }
    _graph->replay();
    ++_replays;
    // Preserve logits retained by the caller across subsequent replays.
    return _output.clone();
  }
  catch (...) {
    _capture_failed = true;
    throw;
  }
#else
  TORCH_CHECK(false, "Decode graph requires an NPU build");
#endif
}

int64_t Qwen3DecodeGraph::capture_count() const {
  return _captures;
}

int64_t Qwen3DecodeGraph::replay_count() const {
  return _replays;
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
