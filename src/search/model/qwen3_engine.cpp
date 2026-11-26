#include "qwen3_engine.h"
#include "qwen3_runner.h"
#include "token/qwen_token.h"
#include <stdexcept>
#include <cstdlib>
#if USE_NPU
#include <torch_npu/torch_npu.h>
#include <acl/acl.h>
#include <acl/acl_prof.h>
#include <filesystem>
#elif USE_GPU
#include <torch/cuda.h>
#endif

#if USE_NPU
static aclprofConfig* init_profile(const std::string& profile_dir, std::string& session_dir) {
  std::filesystem::create_directories(profile_dir);
  // Each collection owns a directory, so export cannot touch earlier sessions.
  auto pattern = (std::filesystem::absolute(profile_dir) / "session-XXXXXX").string();
  TORCH_CHECK(mkdtemp(&pattern[0]) != nullptr, "Cannot create profiling session directory");
  session_dir = pattern;
  int32_t device_id = 0;
  auto ret = aclrtGetDevice(&device_id);
  TORCH_CHECK(ret == ACL_SUCCESS, "aclrtGetDevice failed: ", ret);
  ret = aclprofInit(session_dir.c_str(), session_dir.size());
  TORCH_CHECK(ret == ACL_SUCCESS, "aclprofInit failed: ", ret);
  uint32_t devices[] = {static_cast<uint32_t>(device_id)};
  auto* config = aclprofCreateConfig(devices, 1, ACL_AICORE_NONE, nullptr,
    ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_RUNTIME_API |
    ACL_PROF_AICPU | ACL_PROF_TRAINING_TRACE);
  if (!config) {
    aclprofFinalize();
    TORCH_CHECK(false, "aclprofCreateConfig failed");
  }
  return config;
}
#endif

// Token boundaries can split a UTF-8 character. Hold an incomplete suffix;
// replace malformed bytes so both JSON and SSE always contain valid UTF-8.
static std::string take_text(std::string& pending, bool final) {
  std::string text;
  size_t i = 0;
  while (i < pending.size()) {
    const unsigned char c = pending[i];
    size_t n = c < 0x80 ? 1 : c >= 0xC2 && c <= 0xDF ? 2 :
               c >= 0xE0 && c <= 0xEF ? 3 : c >= 0xF0 && c <= 0xF4 ? 4 : 0;
    if (n && i + n > pending.size() && !final) {
      break;
    }
    bool valid = n && i + n <= pending.size();
    for (size_t j = 1; valid && j < n; ++j) {
      valid = (pending[i + j] & 0xC0) == 0x80;
    }
    if (valid && n >= 3) {
      unsigned char second = pending[i + 1];
      valid = !(c == 0xE0 && second < 0xA0) && !(c == 0xED && second >= 0xA0) &&
              !(c == 0xF0 && second < 0x90) && !(c == 0xF4 && second >= 0x90);
    }
    if (valid) {
      text.append(pending, i, n);
      i += n;
    }
    else {
      text += "\xEF\xBF\xBD";
      ++i;
    }
  }
  pending.erase(0, i);
  return text;
}

Qwen3Engine::Qwen3Engine(
  const std::string& model_path,
  const std::string& device,
  int max_seq_len,
  bool decode_graph,
  const std::string& profile_dir
  ) {
  std::promise<void> ready;
  auto initialized = ready.get_future();
  _thread = std::thread(
    &Qwen3Engine::run,
    this,
    model_path,
    device,
    max_seq_len,
    decode_graph,
    profile_dir,
    std::move(ready)
    );
  try {
    initialized.get();
  }
  catch (...) {
    _thread.join();
    throw;
  }
}

Qwen3Engine::~Qwen3Engine() {
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _stopping = true;
  }
  _condition.notify_one();
  _thread.join();
}

Qwen3Engine::Result Qwen3Engine::generate(const Request& request, TokenCallback on_token) {
  auto job = std::make_shared<Job>();
  job->request = request;
  job->on_token = std::move(on_token);
  return submit(job);
}

void Qwen3Engine::start_profile() {
  auto job = std::make_shared<Job>();
  job->type = Job::Type::StartProfile;
  (void)submit(job);
}

std::string Qwen3Engine::stop_profile() {
  auto job = std::make_shared<Job>();
  job->type = Job::Type::StopProfile;
  return submit(job).profile_dir;
}

Qwen3Engine::Result Qwen3Engine::submit(const std::shared_ptr<Job>& job) {
  auto result = job->result.get_future();
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_stopping) {
      throw std::runtime_error("Engine is stopping");
    }
    _jobs.push(job);
  }
  _condition.notify_one();
  return result.get();
}

void Qwen3Engine::run(
  std::string model_path,
  std::string device,
  int max_seq_len,
  bool decode_graph,
  std::string profile_dir,
  std::promise<void> ready
  ) {
  bool initialized = false;
#if USE_NPU
  aclprofConfig* profile_config = nullptr;
  bool profiling = false;
  std::string session_dir;
#endif
  try {
#if USE_NPU
    if (decode_graph && setenv("TASK_QUEUE_ENABLE", "1", 1) != 0) {
      throw std::runtime_error("Cannot configure NPU queue mode");
    }
    torch_npu::init_npu(device);
    initialized = true;
    // Profiling must be initialized before model loading and graph capture.
    if (!profile_dir.empty()) {
      profile_config = init_profile(profile_dir, session_dir);
    }
    const auto dtype = torch::kBFloat16;
#else
    if (!profile_dir.empty()) {
      throw std::invalid_argument("Profiling requires an NPU build");
    }
    const auto dtype = torch::kHalf;
#endif
    // These objects are destroyed on the inference thread before NPU finalization.
    Qwen3Runner runner(model_path, torch::Device(device), dtype, max_seq_len, decode_graph);
    QwenToken tokenizer(model_path);
    (void)runner.prefill({1});
    if (max_seq_len > 1) {
      (void)runner.decode(1, 1);
    }
    runner.reset();
#if USE_NPU
    torch::npu::synchronize();
#elif USE_GPU
    torch::cuda::synchronize();
#endif
    ready.set_value();
    while (true) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock<std::mutex> lock(_mutex);
        _condition.wait(lock, [this] { return _stopping || !_jobs.empty(); });
        if (_jobs.empty()) {
          break;
        }
        job = _jobs.front();
        _jobs.pop();
      }
      try {
        if (job->type == Job::Type::Generate) {
          job->result.set_value(generate(runner, tokenizer, *job));
        }
        else {
#if USE_NPU
          if (profile_dir.empty()) {
            throw std::invalid_argument("Profiling is disabled; set --profile_dir");
          }
          const bool start = job->type == Job::Type::StartProfile;
          if (start == profiling) {
            throw std::invalid_argument(start ? "Profiling is already active" : "Profiling is not active");
          }
          torch::npu::synchronize();
          if (start && !profile_config) {
            profile_config = init_profile(profile_dir, session_dir);
          }
          const auto ret = start ? aclprofStart(profile_config) : aclprofStop(profile_config);
          TORCH_CHECK(ret == ACL_SUCCESS, start ? "aclprofStart failed: " : "aclprofStop failed: ", ret);
          profiling = start;
          if (!start) {
            const auto destroy_ret = aclprofDestroyConfig(profile_config);
            profile_config = nullptr;
            const auto finalize_ret = aclprofFinalize();
            TORCH_CHECK(destroy_ret == ACL_SUCCESS, "aclprofDestroyConfig failed: ", destroy_ret);
            TORCH_CHECK(finalize_ret == ACL_SUCCESS, "aclprofFinalize failed: ", finalize_ret);
          }
          Result result;
          result.profile_dir = session_dir;
          job->result.set_value(std::move(result));
#else
          throw std::invalid_argument("Profiling requires an NPU build");
#endif
        }
      }
      catch (...) {
        job->result.set_exception(std::current_exception());
      }
    }
  }
  catch (...) {
    ready.set_exception(std::current_exception());
  }
#if USE_NPU
  if (profile_config) {
    if (profiling) {
      const auto ret = aclprofStop(profile_config);
      if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "aclprofStop failed: " << ret;
      }
    }
    const auto ret = aclprofDestroyConfig(profile_config);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "aclprofDestroyConfig failed: " << ret;
    }
    const auto finalize_ret = aclprofFinalize();
    if (finalize_ret != ACL_SUCCESS) {
      LOG(ERROR) << "aclprofFinalize failed: " << finalize_ret;
    }
  }
  if (initialized) {
    try {
      torch_npu::finalize_npu();
    }
    catch (const std::exception& error) {
      LOG(ERROR) << "NPU shutdown: " << error.what();
    }
  }
#endif
}

Qwen3Engine::Result Qwen3Engine::generate(
  Qwen3Runner& runner,
  QwenToken& tokenizer,
  const Job& job
  ) {
  const auto& request = job.request;
  if (request.max_tokens <= 0) {
    throw std::invalid_argument("max_tokens must be positive");
  }
  if (request.prompt.empty()) {
    throw std::invalid_argument("prompt must not be empty");
  }
  // Qwen3 single-turn chat template, enable_thinking=false.
  const std::string prompt = "<|im_start|>user\n" + request.prompt +
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
  auto ids = tokenizer.encode(prompt);
  if (ids.empty() || ids.size() > static_cast<size_t>(runner.capacity()) ||
      request.max_tokens - 1 > runner.capacity() - static_cast<int64_t>(ids.size())) {
    throw std::invalid_argument("prompt and max_tokens exceed max_seq_len, or prompt is empty");
  }
  runner.reset();
  Result result;
  result.prompt_tokens = ids.size();
  result.finish_reason = "length";
  std::vector<int64_t> input(ids.begin(), ids.end());
  auto logits = runner.prefill(input);
  std::string pending;
  for (int step = 0; step < request.max_tokens; ++step) {
    int64_t token = torch::argmax(logits.reshape({-1})).item<int64_t>();
    ++result.completion_tokens;
    if (!request.ignore_eos && (token == 151643 || token == 151645)) {
      result.finish_reason = "stop";
      break;
    }
    pending += tokenizer.decode({static_cast<size_t>(token)});
    auto text = take_text(pending, false);
    result.text += text;
    if (!text.empty() && job.on_token && !job.on_token(text)) {
      result.finish_reason = "cancelled";
      return result;
    }
    if (step + 1 < request.max_tokens) {
      logits = runner.decode(token, ids.size() + step);
    }
  }
  auto text = take_text(pending, true);
  result.text += text;
  if (!text.empty() && job.on_token && !job.on_token(text)) {
    result.finish_reason = "cancelled";
  }
  return result;
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
