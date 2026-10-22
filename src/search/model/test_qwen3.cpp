/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.10.21 14:51:18
 *
\****************************************************/

#include <gtest/gtest.h>
#define C10_USE_GLOG
#include <torch/torch.h>
#include <torch/script.h>
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#if USE_GPU
#include <torch/cuda.h>
#endif
#if USE_NPU
#include <torch_npu/torch_npu.h>
#endif
#include <file_utils.h>
#include <string_utils.h>
#include <timer.h>
#include "utils/model_utils.h"
#include "model/qwen3.h"
#include "token/qwen_token.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr int kRefMantissaBits = 7;
constexpr float kAllowGapUlp = 4.0f;
constexpr float kMaxAbsLimit = 2.0f;

float binary_ulp(float x, int mantissa_bits = kRefMantissaBits) {
  if (!std::isfinite(x)) {
    return std::numeric_limits<float>::infinity();
  }
  x = std::fabs(x);
  if (x == 0.0f) {
    return std::ldexp(1.0f, -mantissa_bits);
  }
  int exp = 0;
  std::frexp(x, &exp);
  return std::ldexp(1.0f, exp - 1 - mantissa_bits);
}

}  // namespace

Qwen3ForCausalLM *_model = NULL;
void InitModel() {
	if (_model) {
    return;
  }
  char *model_name_p = getenv("MODEL_NAME");
  std::string model_name(NULL == model_name_p ? "" : model_name_p);
  auto config = nlohmann::json::parse(std::ifstream("/data/" + model_name + "/config.json"));
  _model = new Qwen3ForCausalLM(
    config["vocab_size"],
    config["hidden_size"],
    config["num_hidden_layers"],
    config["num_attention_heads"],
    config["num_key_value_heads"],
    config["head_dim"],
    config["intermediate_size"],
    config["rms_norm_eps"]
    );
  //load_module_from_safetensors(*_model, "/data/" + model_name + "/model.safetensors");
  load_module_from_safetensors_dir(*_model, "/data/" + model_name);
  if (config["tie_word_embeddings"]) {
    // Exception: a leaf Variable that requires grad is being used in an in-place operation.
    torch::NoGradGuard no_grad;
    _model->lm_head->weight.copy_(_model->model->embed_tokens->weight);
    LOG(INFO) << "Load LM Head Finished";
  }
  {
    torch::NoGradGuard no_grad;
    for (auto& param : _model->parameters()) {
      if (!param.is_contiguous()) {
        param.data() = param.data().contiguous();
      }
    }
  }
  _model->eval();
  LOG(INFO) << "Model Loaded";
}

TEST(Model, Qwen3) {
  char *model_name_p = getenv("MODEL_NAME");
  std::string model_name(NULL == model_name_p ? "" : model_name_p);
  auto config = nlohmann::json::parse(
  std::ifstream("/data/" + model_name + "/config.json"));
  int vocab_size = config["vocab_size"];
  int hidden_size = config["hidden_size"];
  int num_layers = config["num_hidden_layers"];
  int num_heads = config["num_attention_heads"];
  int num_kv_heads = config["num_key_value_heads"];
  int head_dim = config["head_dim"];
  int intermediate_size = config["intermediate_size"];
  float rms_norm_eps = config["rms_norm_eps"];
  bool tie_word_embeddings = config.value("tie_word_embeddings", false);

  Qwen3ForCausalLM model(
    vocab_size,
    hidden_size,
    num_layers,
    num_heads,
    num_kv_heads,
    head_dim,
    intermediate_size,
    rms_norm_eps
    );
  //load_module_from_safetensors(model, "/data/" + model_name + "/model.safetensors");
  load_module_from_safetensors_dir(model, "/data/" + model_name);
  {
    torch::NoGradGuard no_grad;
    for (auto& param : model.parameters()) {
      if (!param.is_contiguous()) {
        param.data() = param.data().contiguous();
      }
    }
  }
  model.eval();
  for (const auto& p : model.named_parameters()) {
    LOG(INFO) << "Key[" << p.key() << "] Shape[" << p.value().sizes() << "]";
  }
}

TEST(Model, Qwen3ModelEmb) {
  InitModel();
  
  std::string content;
  mycommon::file_read("./test_embed.pt", content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	auto ids = data.at("ids").toTensor();
	auto ref = data.at("ref").toTensor();
	auto weight = data.at("weight").toTensor();

  auto out = _model->model->embed_tokens->forward(ids);
  double rtol = 1e-5;
	double atol = 1e-5;
	ASSERT_EQ(true, check_close("out", out, ref, rtol, atol));
}

TEST(Model, Qwen3ModelLayerInputLayer0) {
  InitModel();
  
  std::string content;
  mycommon::file_read("./test_input_layernorm_0.pt", content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	auto ids = data.at("ids").toTensor();
	auto input = data.at("input").toTensor();
	auto ref = data.at("ref").toTensor();
  
  auto out = ((Qwen3DecoderLayer *)_model->model->layers[0].get())->input_layernorm->forward(input);
  double rtol = 1e-5;
	double atol = 1e-5;
	ASSERT_EQ(true, check_close("out", out, ref, rtol, atol));
}

TEST(Model, Qwen3ModelLayerAttn0) {
  InitModel();
  
  std::string content;
  mycommon::file_read("./test_self_attn_0.pt", content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	auto ids = data.at("ids").toTensor();
	auto hidden_states = data.at("hidden_states").toTensor();
	auto position_embeddings = std::make_tuple(data.at("position_embeddings_cos").toTensor(), data.at("position_embeddings_sin").toTensor());
	auto attention_mask = data.at("attention_mask").toTensor();
	auto ref_output_attn_output = data.at("output_attn_output").toTensor();
	auto ref_output_attn_weights = data.at("output_attn_weights").toTensor();
	
  auto [attn_output, attn_weights]  = ((Qwen3DecoderLayer *)_model->model->layers[0].get())->self_attn->forward(hidden_states, position_embeddings, attention_mask);
  double rtol = 1e-5;
	double atol = 1e-5;
	ASSERT_EQ(true, check_close("attn_output", attn_output, ref_output_attn_output, rtol, atol));
	ASSERT_EQ(true, check_close("attn_weights", attn_weights, ref_output_attn_weights, rtol, atol));
}

TEST(Model, Qwen3ModelLayer0) {
  InitModel();
  
  std::string content;
  mycommon::file_read("./test_layer_0.pt", content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	auto ids = data.at("ids").toTensor();
	auto hidden_states = data.at("norm_input").toTensor();
	auto position_embeddings = std::make_tuple(data.at("position_embeddings_cos").toTensor(), data.at("position_embeddings_sin").toTensor());
	auto attention_mask = data.at("attention_mask").toTensor();
	auto ref_attn_output = data.at("attn_output").toTensor();
	auto ref_layer_output = data.at("layer_output").toTensor();
  
  //LOG(INFO) << "Test layer_norm Input [" << format_tensor(data.at("norm_input").toTensor()) << "]";
  //LOG(INFO) << "Test layer_norm Output [" << format_tensor(data.at("norm_output").toTensor()) << "]";
  //LOG(INFO) << "Test self_attn hidden_states [" << format_tensor(data.at("hidden_states").toTensor()) << "]";
  //LOG(INFO) << "Test self_attn position_embeddings_cos [" << format_tensor(data.at("position_embeddings_cos").toTensor()) << "]";
  //LOG(INFO) << "Test self_attn position_embeddings_sin [" << format_tensor(data.at("position_embeddings_sin").toTensor()) << "]";
  //LOG(INFO) << "Test self_attn attention_mask [" << format_tensor(data.at("attention_mask").toTensor()) << "]";
  //LOG(INFO) << "Test self_attn attn_output [" << format_tensor(data.at("attn_output").toTensor()) << "]";
  //LOG(INFO) << "Test self_attn attn_weights [" << format_tensor(data.at("attn_weights").toTensor()) << "]";
  //LOG(INFO) << "Test post_norm Input [" << format_tensor(data.at("post_norm_input").toTensor()) << "]";
  //LOG(INFO) << "Test post_norm Output [" << format_tensor(data.at("post_norm_output").toTensor()) << "]";
  
  auto layer_output = ((Qwen3DecoderLayer *)_model->model->layers[0].get())->forward(hidden_states, position_embeddings, attention_mask);
  //LOG(INFO) << "Out[" << format_tensor(layer_output) << "]";
  double rtol = 1e-5;
	double atol = 1e-5;
	ASSERT_EQ(true, check_close("layer_output", layer_output, ref_layer_output, rtol, atol));
}

TEST(Model, Qwen3ModelLayer1) {
  InitModel();
  
  std::string content;
  mycommon::file_read("./test_layer_0.pt", content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	auto ids = data.at("ids").toTensor();
	auto hidden_states = data.at("norm_input").toTensor();
	auto position_embeddings = std::make_tuple(data.at("position_embeddings_cos").toTensor(), data.at("position_embeddings_sin").toTensor());
	auto attention_mask = data.at("attention_mask").toTensor();
	auto ref_attn_output_0 = data.at("attn_output").toTensor();
	auto ref_layer_output_0 = data.at("layer_output").toTensor();
  
  mycommon::file_read("./test_layer_1.pt", content);
  torch::IValue ivalue_1 = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue_1.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data_1 = ivalue_1.toGenericDict();
	auto ref_layer_output_1 = data_1.at("layer_output").toTensor();
  
  auto layer_output_0 = ((Qwen3DecoderLayer *)_model->model->layers[0].get())->forward(hidden_states, position_embeddings, attention_mask);
  auto layer_output_1 = ((Qwen3DecoderLayer *)_model->model->layers[1].get())->forward(layer_output_0, position_embeddings, attention_mask);
  
  double rtol = 1e-5;
	double atol = 1e-5;
	ASSERT_EQ(true, check_close("layer_output", layer_output_1, ref_layer_output_1, rtol, atol));
}

TEST(Model, Qwen3RotaryEmb) {
  InitModel();
  
  std::string content;
  mycommon::file_read("./test_rotary_emb.pt", content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	auto ids = data.at("ids").toTensor();
	auto hidden_states = data.at("hidden_states").toTensor();
	auto position_ids = data.at("position_ids").toTensor();
	auto ref_output_cos = data.at("output_cos").toTensor();
	auto ref_output_sin = data.at("output_sin").toTensor();
  
  auto [output_cos, output_sin] = (_model->model->rotary_emb.get())->forward(hidden_states, position_ids);
  double rtol = 1e-5;
	double atol = 1e-5;
	ASSERT_EQ(true, check_close("output_cos", output_cos, ref_output_cos, rtol, atol));
	ASSERT_EQ(true, check_close("output_sin", output_sin, ref_output_sin, rtol, atol));
}

TEST(Model, Qwen3AttentionMask) {
  InitModel();
  
  std::string content;
  mycommon::file_read("./test_rotary_emb.pt", content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	auto ids = data.at("ids").toTensor();
	auto ref_hidden_states = data.at("hidden_states").toTensor();
	auto ref_position_ids = data.at("position_ids").toTensor();

  std::vector<int64_t> input_ids(ids.data_ptr<int64_t>(), ids.data_ptr<int64_t>() + ids.numel());
  auto [hidden_states, position_ids] = _model->model->prepare(input_ids);
  double rtol = 1e-5;
	double atol = 1e-5;
	ASSERT_EQ(true, check_close("hidden_states", hidden_states, ref_hidden_states, rtol, atol));
	ASSERT_EQ(true, check_close("position_ids", position_ids, ref_position_ids, rtol, atol));
}

TEST(Model, Qwen3Model) {
  InitModel();
  
  std::string content;
  mycommon::file_read("./test_model.pt", content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	auto ids = data.at("ids").toTensor();
	auto ref_last_hidden_state = data.at("last_hidden_state").toTensor();

  std::vector<int64_t> input_ids(ids.data_ptr<int64_t>(), ids.data_ptr<int64_t>() + ids.numel());
  auto last_hidden_state = _model->model->forward(input_ids);
  double rtol = 1e-4;
	double atol = 1e-4;
	ASSERT_EQ(true, check_close("last_hidden_state", last_hidden_state, ref_last_hidden_state, rtol, atol));
}

TEST(Model, Qwen3Lm) {
  InitModel();
  
  std::string content;
  mycommon::file_read("./test_lm.pt", content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	auto ids = data.at("ids").toTensor();
	auto ref_logits = data.at("logits").toTensor();

  std::vector<int64_t> input_ids(ids.data_ptr<int64_t>(), ids.data_ptr<int64_t>() + ids.numel());
  auto logits = _model->forward(input_ids, 0);
  double rtol = 1e-4;
	double atol = 1e-4;
	ASSERT_EQ(true, check_close("logits", logits, ref_logits, rtol, atol));
}

TEST(Model, Qwen3Runner) {
  InitModel();
  char *model_name_p = getenv("MODEL_NAME");
  std::string model_name(NULL == model_name_p ? "" : model_name_p);
  
  std::string content;
  mycommon::file_read("./test_runner.pt", content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
	auto data = ivalue.toGenericDict();
	auto ids = data.at("ids").toTensor();
	auto max_new_tokens = data.at("max_length").toInt();
	auto ref_output = data.at("output").toTensor();
  std::vector<int64_t> ref_output_ids(ref_output.data_ptr<int64_t>(), ref_output.data_ptr<int64_t>() + ref_output.numel());
  
  std::vector<int64_t> ref_input_ids(ids.data_ptr<int64_t>(), ids.data_ptr<int64_t>() + ids.numel());
  //auto logits = _model->forward(ref_input_ids);
  //auto token_ids_tensor = torch::argmax(logits, /*dim=*/-1);
  ////std::vector<int64_t> token_ids(token_ids_tensor.data_ptr<int64_t>(), token_ids_tensor.data_ptr<int64_t>() + token_ids_tensor.numel());
  //double rtol = 1e-4;
	//double atol = 1e-4;
	//ASSERT_EQ(true, check_close("token_ids", token_ids_tensor, ref_output, rtol, atol));
  
  uint64_t time_start = mycommon::getMilliTime();
  std::vector<int64_t> current_tokens = ref_input_ids;
  for (int step = 0; step < max_new_tokens - 1; ++step) {
    //auto input_ids = torch::from_blob(
    //  current_tokens.data(), 
    //  {1, static_cast<int64_t>(current_tokens.size())}, 
    //  torch::kInt64
    //).clone();
    //auto logits = _model->forward(input_ids, /*logits_to_keep=*/1);
    auto logits = _model->forward(current_tokens, /*logits_to_keep=*/1);

    // 贪心
    auto next_token_tensor = torch::argmax(logits.squeeze(0).squeeze(0), /*dim=*/-1);
    int64_t next_token = next_token_tensor.item<int64_t>();

    current_tokens.push_back(next_token);

    LOG(INFO) << "Step[" << step << "] Token[" << next_token << "]";

    if (next_token == 151643 || next_token == 151645) {
      LOG(INFO) << "EOS End";
      break;
    }
  }
  uint64_t time_used = mycommon::getMilliTime() - time_start;
  LOG(INFO) << "Time[" << time_used << "] Input[" << ref_input_ids.size() << "] Output[" << current_tokens.size() << "]";
  QwenToken token("/data/" + model_name);
  std::vector<size_t> run_ids;
  for (auto _ : current_tokens) {
    run_ids.push_back(_);
  }
  std::string my_text = token.decode(run_ids);
  LOG(INFO) << my_text;
  ASSERT_EQ(current_tokens, ref_output_ids);
}

inline void sync_device() {
#if USE_GPU
  torch::cuda::synchronize();
#elif USE_NPU
  torch::npu::synchronize();
#endif
}

int64_t greedy_token_id(const torch::Tensor& logits) {
  return torch::argmax(logits.reshape({-1})).item<int64_t>();
}

class Qwen3RunnerCache : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    LOG(INFO) << "num_threads: " << torch::get_num_threads();
    LOG(INFO) << "num_interop_threads: " << torch::get_num_interop_threads();
    LOG(INFO) << "MKL available: " << at::hasMKL();
    LOG(INFO) << "MKL-DNN available: " << at::hasMKLDNN();
    char *model_name_p = getenv("MODEL_NAME");
    model_name = (NULL == model_name_p ? "" : model_name_p);

#if USE_NPU
    LOG(INFO) << "Init NPU...";
    torch_npu::init_npu("npu:0");
    device = at::Device("npu:0");
    LOG(INFO) << "Init NPU Finished";
#elif USE_GPU
    device = torch::Device(torch::kCUDA);
#else
    device = torch::Device(torch::kCPU);
#endif

    auto config = nlohmann::json::parse(std::ifstream("/data/" + model_name + "/config.json"));
    int max_seq_len = 2048;
    int batch_size = 1;
    LOG(INFO) << "Init KV Cache...";
    kv_cache = new KVCache(
      config["num_hidden_layers"],
      batch_size,
      config["num_key_value_heads"],
      config["head_dim"],
      max_seq_len,
#if USE_GPU
      torch::kHalf,
      torch::kCUDA
#elif USE_NPU
      torch::kBFloat16,
      device
#else
      torch::kHalf,
      torch::kCPU
#endif
      );
    LOG(INFO) << "Init KV Cache Finished";
#if USE_GPU
    auto tensor_options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
#elif USE_NPU
    auto tensor_options = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
#else
    auto tensor_options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCPU);
#endif
    model = new Qwen3ForCausalLM(
      config["vocab_size"],
      config["hidden_size"],
      config["num_hidden_layers"],
      config["num_attention_heads"],
      config["num_key_value_heads"],
      config["head_dim"],
      config["intermediate_size"],
      config["rms_norm_eps"],
      kv_cache,
      tensor_options
      );
    LOG(INFO) << "Init Qwen3ForCausalLM Finished";
    load_module_from_safetensors_dir(*model, "/data/" + model_name);
    LOG(INFO) << "Load Model Tensors Finished";
    if (config["tie_word_embeddings"]) {
      torch::NoGradGuard no_grad;
      model->lm_head->weight.copy_(model->model->embed_tokens->weight);
      LOG(INFO) << "Load LM Head Finished";
    }
    {
      torch::NoGradGuard no_grad;
      for (auto& param : model->parameters()) {
        if (!param.is_contiguous()) {
          param.data() = param.data().contiguous();
        }
      }
      LOG(INFO) << "Contiguous Finished";
    }
    model->eval();
#if USE_GPU
    model->to(torch::kCUDA);
    model->to(torch::kHalf);
#elif USE_NPU
    model->to(device);
    model->to(torch::kBFloat16);
#else
    model->to(torch::kHalf);
#endif
    LOG(INFO) << "Model Loaded";
  }

  static void TearDownTestSuite() {
    delete model;
    model = nullptr;
    delete kv_cache;
    kv_cache = nullptr;
#if USE_NPU
    torch_npu::finalize_npu();
#endif
  }

  void SetUp() override {
    ASSERT_NE(kv_cache, nullptr);
    kv_cache->reset();
  }

  static std::string model_name;
  static torch::Device device;
  static KVCache *kv_cache;
  static Qwen3ForCausalLM *model;
};

std::string Qwen3RunnerCache::model_name;
torch::Device Qwen3RunnerCache::device = torch::Device(torch::kCPU);
KVCache *Qwen3RunnerCache::kv_cache = nullptr;
Qwen3ForCausalLM *Qwen3RunnerCache::model = nullptr;

TEST_F(Qwen3RunnerCache, Correctness) {
  std::string content;
  mycommon::file_read("./test_runner.pt", content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
  auto data = ivalue.toGenericDict();
  auto ids = data.at("ids").toTensor();
  auto max_new_tokens = data.at("max_length").toInt();
  auto ref_output = data.at("output").toTensor();
  std::vector<int64_t> ref_output_ids(ref_output.data_ptr<int64_t>(), ref_output.data_ptr<int64_t>() + ref_output.numel());
  std::vector<int64_t> ref_input_ids(ids.data_ptr<int64_t>(), ids.data_ptr<int64_t>() + ids.numel());
  ASSERT_TRUE(mycommon::file_exist("./test_runner_logits.pt")) << "Run test_runner.py to generate reference logits";
  mycommon::file_read("./test_runner_logits.pt", content);
  auto logits_data = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end())).toGenericDict();
  auto ref_logits = logits_data.at("logits").toTensor();
  ASSERT_TRUE(torch::equal(logits_data.at("output").toTensor(), ref_output)) << "Reference logits are stale";
  ASSERT_EQ(ref_logits.dim(), 2);
  ASSERT_EQ(ref_logits.size(0), ref_output_ids.size() - ref_input_ids.size());
  auto config = nlohmann::json::parse(std::ifstream("/data/" + model_name + "/config.json"));
  ASSERT_EQ(ref_logits.size(1), config["vocab_size"].get<int64_t>());
  const int64_t prompt_len = static_cast<int64_t>(ref_input_ids.size());
  const int64_t gen_len = std::min(
    static_cast<int64_t>(max_new_tokens),
    static_cast<int64_t>(ref_output_ids.size()) - prompt_len);
  ASSERT_EQ(gen_len, ref_logits.size(0));

  std::vector<int64_t> greedy_tokens = ref_input_ids;
  float max_abs_err = 0.0f;
  float mean_abs_sum = 0.0f;
  float max_large_ulp_err = 0.0f;
  int64_t max_abs_step = -1;
  int64_t max_ulp_step = -1;
  int match_count = 0;
  int allowed_mismatch = 0;
  int hard_mismatch = 0;
  int64_t first_mismatch_step = -1;

  auto eval_step = [&](int64_t step, const torch::Tensor& step_logits) {
    const int64_t expected = ref_output_ids[prompt_len + step];
    auto scores = step_logits.reshape({-1}).to(torch::kCPU).to(torch::kFloat32);
    auto ref_scores = ref_logits[step].to(torch::kFloat32);
    const int64_t selected = torch::argmax(scores).item<int64_t>();
    greedy_tokens.push_back(selected);

    auto diff = (scores - ref_scores).abs();
    const float step_max_abs = diff.max().item<float>();
    const float step_mean_abs = diff.mean().item<float>();
    mean_abs_sum += step_mean_abs;
    if (step_max_abs > max_abs_err) {
      max_abs_err = step_max_abs;
      max_abs_step = step;
    }

    const float ref_peak = ref_scores.abs().max().item<float>();
    const float step_max_ulp = step_max_abs / binary_ulp(std::max(ref_peak, 1.0f));
    if (step_max_ulp > max_large_ulp_err) {
      max_large_ulp_err = step_max_ulp;
      max_ulp_step = step;
    }

    auto [ref_topk_values, ref_topk_indices] = torch::topk(ref_scores, 2);
    const float ref_gap = (ref_topk_values[0] - ref_topk_values[1]).item<float>();
    const float ref_gap_ulp = ref_gap / binary_ulp(ref_topk_values[0].item<float>());
    const int64_t ref_top1 = ref_topk_indices[0].item<int64_t>();
    const int64_t ref_top2 = ref_topk_indices[1].item<int64_t>();
    const bool in_ref_top2 = selected == ref_top1 || selected == ref_top2;
    const bool mismatch = selected != expected;
    if (!mismatch) {
      ++match_count;
      return selected;
    }
    if (first_mismatch_step < 0) {
      first_mismatch_step = step;
    }
    const bool allowed = in_ref_top2 && ref_gap_ulp <= kAllowGapUlp;
    if (allowed) {
      ++allowed_mismatch;
    } else {
      ++hard_mismatch;
    }
    auto [actual_topk_values, actual_topk_indices] = torch::topk(scores, 2);
    LOG(INFO) << (allowed ? "Allowed" : "Hard")
              << " mismatch step[" << step << "] expected[" << expected
              << "] selected[" << selected << "] in_ref_top2[" << in_ref_top2
              << "] ref_gap[" << ref_gap << "] ref_gap_ulp[" << ref_gap_ulp << "]";
    LOG(INFO) << "Reference Top2[" << ref_top1 << ":"
              << ref_topk_values[0].item<float>() << ", " << ref_top2 << ":"
              << ref_topk_values[1].item<float>() << "]";
    LOG(INFO) << "Actual Top2[" << actual_topk_indices[0].item<int64_t>() << ":"
              << actual_topk_values[0].item<float>() << ", "
              << actual_topk_indices[1].item<int64_t>() << ":"
              << actual_topk_values[1].item<float>() << "]";
    LOG(INFO) << "Candidate delta expected["
              << scores[expected].item<float>() - ref_scores[expected].item<float>()
              << "] selected["
              << scores[selected].item<float>() - ref_scores[selected].item<float>()
              << "]";
    return selected;
  };

  auto logits = model->forward(
    ref_input_ids,
    1,
    0
    );
  eval_step(0, logits);
  int64_t feed = ref_output_ids[prompt_len];
  for (int64_t step = 1; step < gen_len; ++step) {
    if (feed == 151643 || feed == 151645) {
      LOG(INFO) << "EOS End at teacher-force step[" << step - 1 << "]";
      break;
    }
    logits = model->forward({feed}, /*logits_to_keep=*/1, prompt_len + step - 1);
    eval_step(step, logits);
    feed = ref_output_ids[prompt_len + step];
  }
  const int eval_steps = match_count + allowed_mismatch + hard_mismatch;
  const float mean_abs_err = mean_abs_sum / std::max(eval_steps, 1);
  LOG(INFO) << "Eval match[" << match_count
            << "] allowed_mismatch[" << allowed_mismatch
            << "] hard_mismatch[" << hard_mismatch
            << "] first_mismatch_step[" << first_mismatch_step
            << "] max_abs[" << max_abs_err << "]@step[" << max_abs_step
            << "] mean_abs[" << mean_abs_err
            << "] max_ulp_at_peak[" << max_large_ulp_err << "]@step[" << max_ulp_step
            << "] allow_gap_ulp[" << kAllowGapUlp << "]";
  QwenToken token("/data/" + model_name);
  std::vector<size_t> run_ids;
  for (auto id : greedy_tokens) {
    run_ids.push_back(static_cast<size_t>(id));
  }
  std::vector<size_t> ref_run_ids;
  for (auto id : ref_output_ids) {
    ref_run_ids.push_back(static_cast<size_t>(id));
  }
  LOG(INFO) << "Ori[" << token.decode(ref_run_ids) << "]";
  LOG(INFO) << "Our[" << token.decode(run_ids) << "]";
  LOG(INFO) << "Ori[" << mycommon::str_join(ref_output_ids) << "]";
  LOG(INFO) << "Our[" << mycommon::str_join(greedy_tokens) << "]";
  EXPECT_EQ(hard_mismatch, 0) << "Greedy argmax left reference top2 while top2 gap was > "
                              << kAllowGapUlp << " ULP";
  EXPECT_LE(max_abs_err, kMaxAbsLimit) << "Teacher-force logits max abs error too large";
}

TEST_F(Qwen3RunnerCache, Perf) {
  std::string content;
  mycommon::file_read("./test_runner.pt", content);
  torch::IValue ivalue = torch::jit::pickle_load(std::vector<char>(content.begin(), content.end()));
  if (!ivalue.isGenericDict()) {
    LOG(WARNING) << "Loaded data is not a dictionary!";
    ASSERT_EQ(true, false);
  }
  auto data = ivalue.toGenericDict();
  auto ids = data.at("ids").toTensor();
  auto max_new_tokens = data.at("max_length").toInt();
  auto ref_output = data.at("output").toTensor();
  std::vector<int64_t> ref_output_ids(ref_output.data_ptr<int64_t>(), ref_output.data_ptr<int64_t>() + ref_output.numel());
  std::vector<int64_t> ref_input_ids(ids.data_ptr<int64_t>(), ids.data_ptr<int64_t>() + ids.numel());
  const int64_t prompt_len = static_cast<int64_t>(ref_input_ids.size());
  const int64_t gen_len = std::min(
    static_cast<int64_t>(max_new_tokens),
    static_cast<int64_t>(ref_output_ids.size()) - prompt_len);
  ASSERT_GT(gen_len, 0);

  {
    auto warmup_logits = model->forward(ref_input_ids, 1, 0);
    const int64_t warmup_token = greedy_token_id(warmup_logits);
    (void)model->forward({warmup_token}, 1, prompt_len);
    sync_device();
    kv_cache->reset();
  }

  sync_device();
  uint64_t time_start = mycommon::getMilliTime();
  auto logits = model->forward(
    ref_input_ids,
    1,
    0
    );
  sync_device();
  uint64_t time_prefill = mycommon::getMilliTime() - time_start;

  std::vector<int64_t> greedy_tokens = ref_input_ids;
  time_start = mycommon::getMilliTime();
  for (int64_t step = 0; step < gen_len; ++step) {
    const int64_t next_token = greedy_token_id(logits);
    greedy_tokens.push_back(next_token);
    if (step + 1 >= gen_len) {
      break;
    }
    logits = model->forward({next_token}, /*logits_to_keep=*/1, prompt_len + step);
  }
  sync_device();
  uint64_t time_decode = mycommon::getMilliTime() - time_start;
  const int64_t gen_tokens = static_cast<int64_t>(greedy_tokens.size()) - prompt_len;
  LOG(INFO) << "TTFT[" << time_prefill << "] "
            << "Decode Time[" << time_decode << "] "
            << "TPOT[" << ((float)time_decode / std::max(gen_tokens, static_cast<int64_t>(1))) << "] "
            << "PrefillPerf[" << ((float)prompt_len * 1000 / time_prefill) << "] "
            << "OutPerf[" << ((float)gen_tokens * 1000 / time_decode) << "] "
            << "Input[" << prompt_len << "] "
            << "Output[" << gen_tokens << "]";
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
