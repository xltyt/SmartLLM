#include "qwen3_runner.h"
#include "utils/model_utils.h"
#include <nlohmann/json.hpp>
#include <fstream>

Qwen3Runner::Qwen3Runner(const std::string& model_path, torch::Device device,
                         torch::Dtype dtype, int max_seq_len, bool decode_graph) {
  TORCH_CHECK(max_seq_len > 0, "Runner max sequence length must be positive");
  TORCH_CHECK(!decode_graph || (Qwen3DecodeGraph::supported() && device.is_privateuseone()),
              "Decode graph requires an NPU build and device");
  std::ifstream config_file(model_path + "/config.json");
  TORCH_CHECK(config_file.is_open(), "Cannot open model config: ", model_path, "/config.json");
  auto config = nlohmann::json::parse(config_file);
  _vocab_size = config["vocab_size"].get<int64_t>();
  _kv_cache = std::make_unique<KVCache>(
    config["num_hidden_layers"],
    1,
    config["num_key_value_heads"],
    config["head_dim"],
    max_seq_len,
    dtype,
    device
    );
  auto options = torch::TensorOptions().dtype(dtype).device(device);
  _model = std::make_unique<Qwen3ForCausalLM>(
    config["vocab_size"],
    config["hidden_size"],
    config["num_hidden_layers"],
    config["num_attention_heads"],
    config["num_key_value_heads"],
    config["head_dim"],
    config["intermediate_size"],
    config["rms_norm_eps"],
    _kv_cache.get(),
    options
    );
  load_module_from_safetensors_dir(*_model, model_path);
  {
    torch::NoGradGuard no_grad;
    if (config["tie_word_embeddings"]) {
      _model->lm_head->weight.copy_(_model->model->embed_tokens->weight);
    }
    for (auto& param : _model->parameters()) {
      if (!param.is_contiguous()) {
        param.data() = param.data().contiguous();
      }
    }
  }
  _model->eval();
  _model->to(device);
  _model->to(dtype);
  _model->prepare_fused_weights();
  auto input_options = torch::TensorOptions().dtype(torch::kInt64).device(device);
  _decode_token = torch::zeros({1, 1}, input_options);
  _decode_position = torch::zeros({1}, input_options);
  if (decode_graph) {
    _decode_graph = std::make_unique<Qwen3DecodeGraph>(*_model, *_kv_cache, capacity());
  }
}

torch::Tensor Qwen3Runner::prefill(const std::vector<int64_t>& input_ids) {
  TORCH_CHECK(!input_ids.empty() && input_ids.size() <= static_cast<size_t>(capacity()),
              "Prompt must be nonempty and fit in the KV cache");
  for (int64_t token : input_ids) {
    TORCH_CHECK(token >= 0 && token < _vocab_size, "Prompt token is outside the vocabulary");
  }
  return _model->forward(input_ids, 1, 0);
}

torch::Tensor Qwen3Runner::decode(int64_t token_id, int64_t position) {
  TORCH_CHECK(position >= 0 && position < capacity(), "Decode position exceeds KV capacity");
  TORCH_CHECK(token_id >= 0 && token_id < _vocab_size, "Decode token is outside the vocabulary");
  if (_decode_graph) {
    return _decode_graph->forward(token_id, position);
  }
#if USE_GPU
  return _model->forward({token_id}, 1, position);
#else
  _decode_token.fill_(token_id);
  _decode_position.fill_(position);
  DecodeContext context{_decode_position, capacity()};
  return _model->forward_static_decode(_decode_token, context);
#endif
}

void Qwen3Runner::reset() {
  _kv_cache->reset();
}

int64_t Qwen3Runner::capture_count() const {
  return _decode_graph ? _decode_graph->capture_count() : 0;
}

int64_t Qwen3Runner::replay_count() const {
  return _decode_graph ? _decode_graph->replay_count() : 0;
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
