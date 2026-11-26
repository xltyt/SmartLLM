#ifndef QWEN3_ENGINE_H
#define QWEN3_ENGINE_H

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

class Qwen3Runner;
class QwenToken;

class Qwen3Engine {
public:
  struct Request {
    std::string prompt;
    int max_tokens = 128;
    bool ignore_eos = false;
  };
  struct Result {
    std::string text;
    std::string finish_reason;
    std::string profile_dir;
    int prompt_tokens = 0;
    int completion_tokens = 0;
  };
  // Return false when the consumer disconnects, to stop generation.
  using TokenCallback = std::function<bool(const std::string&)>;

  Qwen3Engine(
    const std::string& model_path,
    const std::string& device,
    int max_seq_len,
    bool decode_graph,
    const std::string& profile_dir = ""
    );
  ~Qwen3Engine();
  Result generate(const Request& request, TokenCallback on_token = {});
  void start_profile();
  std::string stop_profile();

private:
  struct Job {
    enum class Type { Generate, StartProfile, StopProfile };
    Type type = Type::Generate;
    Request request;
    TokenCallback on_token;
    std::promise<Result> result;
  };
  Result submit(const std::shared_ptr<Job>& job);
  void run(
    std::string model_path,
    std::string device,
    int max_seq_len,
    bool decode_graph,
    std::string profile_dir,
    std::promise<void> ready
    );
  Result generate(
    Qwen3Runner& runner,
    QwenToken& tokenizer,
    const Job& job
    );

  std::mutex _mutex;
  std::condition_variable _condition;
  std::queue<std::shared_ptr<Job>> _jobs;
  bool _stopping = false;
  std::thread _thread;
};

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
