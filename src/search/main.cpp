/****************************************************\
 *
 * Copyright (C) 2025 All Rights Reserved
 * Last modified: 2026.07.20 15:52:37
 *
\****************************************************/

#include <bvar/bvar.h>
#include <brpc/server.h>
#include <brpc/restful.h>
#include <glog/logging.h>
#include <search.pb.h>
#include <common.h>
#include <base64.h>
#include "flags.h"
#include "model/qwen3_engine.h"
#include <atomic>
#include <ctime>

#define APP_VERSION "0.1"

int GetUriQueryInt(const brpc::URI& uri, const std::string&key, int default_val) {
  const std::string *param_val = uri.GetQuery(key);
  int val = (NULL == param_val ? default_val : atoi(param_val->c_str()));
  return val;
}

uint64_t GetUriQueryUInt(const brpc::URI& uri, const std::string&key, int default_val) {
  const std::string *param_val = uri.GetQuery(key);
  uint64_t val = (NULL == param_val ? default_val : strtoul(param_val->c_str(), NULL, 10));
  return val;
}

bool GetUriQueryBool(const brpc::URI& uri, const std::string&key, bool default_val) {
  const std::string *param_val = uri.GetQuery(key);
  bool val = (NULL == param_val ? default_val : (bool)atol(param_val->c_str()));
  return val;
}

std::string GetUriQueryString(const brpc::URI& uri, const std::string&key, const std::string& default_val) {
  const std::string *param_val = uri.GetQuery(key);
  return std::string(NULL == param_val ? default_val : param_val->c_str());
}

std::string GenRespBody(int _status, const std::string& _error, const std::string& _data = "") {
  rapidjson::Document doc_result;
  rapidjson::Document::AllocatorType& allocator_result = doc_result.GetAllocator();
  rapidjson::Value json_val_result(rapidjson::kObjectType);
  JSON_ADD_INT(json_val_result, status, "status", _status);
  if (0 != _status) {
    JSON_ADD_STRING(json_val_result, error, "error", _error);
  }
  else {
    JSON_ADD_STRING(json_val_result, data, "data", _data);
  }
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
  json_val_result.Accept(writer);
  return sb.GetString();
}

static std::string CompletionJson(const std::string& id, uint64_t created,
                                  const std::string& text, const Qwen3Engine::Result* result) {
  rapidjson::Document doc_result;
  auto& allocator_result = doc_result.GetAllocator();
  rapidjson::Value root(rapidjson::kObjectType), choice(rapidjson::kObjectType);
  rapidjson::Value choices(rapidjson::kArrayType);
  JSON_ADD_STRING(root, id, "id", id);
  JSON_ADD_STRING(root, object, "object", std::string("text_completion"));
  JSON_ADD_UINT(root, created, "created", created);
  JSON_ADD_STRING(root, model, "model", FLAGS_model_path);
  JSON_ADD_STRING(choice, text, "text", text);
  JSON_ADD_INT(choice, index, "index", 0);
  choice.AddMember("logprobs", rapidjson::Value(), allocator_result);
  if (result) {
    JSON_ADD_STRING(choice, finish, "finish_reason", result->finish_reason);
  }
  else {
    choice.AddMember("finish_reason", rapidjson::Value(), allocator_result);
  }
  choices.PushBack(choice, allocator_result);
  root.AddMember("choices", choices, allocator_result);
  if (result) {
    rapidjson::Value usage(rapidjson::kObjectType);
    JSON_ADD_INT(usage, prompt_tokens, "prompt_tokens", result->prompt_tokens);
    JSON_ADD_INT(usage, completion_tokens, "completion_tokens", result->completion_tokens);
    JSON_ADD_INT(usage, total_tokens, "total_tokens", result->prompt_tokens + result->completion_tokens);
    root.AddMember("usage", usage, allocator_result);
  }
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  root.Accept(writer);
  return std::string(buffer.GetString(), buffer.GetSize());
}

static std::string CompletionError(const std::string& message, bool invalid) {
  rapidjson::Document doc_result;
  auto& allocator_result = doc_result.GetAllocator();
  rapidjson::Value root(rapidjson::kObjectType), error(rapidjson::kObjectType);
  JSON_ADD_STRING(error, message, "message", message);
  JSON_ADD_STRING(error, type, "type", std::string(invalid ? "invalid_request_error" : "server_error"));
  error.AddMember("param", rapidjson::Value(), allocator_result);
  error.AddMember("code", rapidjson::Value(), allocator_result);
  root.AddMember("error", error, allocator_result);
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  root.Accept(writer);
  return std::string(buffer.GetString(), buffer.GetSize());
}

static void Daemonize() {
  if (fork() > 0) {
    exit(0);
  }
  setsid();
  close(STDIN_FILENO);
  close(STDOUT_FILENO);
  close(STDERR_FILENO);
}

class MergeServiceImpl : public proto::search {
public:
  explicit MergeServiceImpl(Qwen3Engine& engine)
    : _engine(engine) {
  }
  virtual ~MergeServiceImpl() {
  }
  void default_method(google::protobuf::RpcController *cntl_base, const proto::SearchRequest *, proto::SearchResponse *, google::protobuf::Closure *done) {
    brpc::ClosureGuard done_guard(done);
    brpc::Controller *cntl = static_cast<brpc::Controller *>(cntl_base);
    brpc::HttpHeader& header = cntl->http_request();
    std::string unresolved_path = header.unresolved_path();
    //std::string uri_path = header.uri().path();
    std::ostringstream oss;
    header.uri().PrintWithoutHost(oss);
    std::string uri_path = oss.str();
    std::string client_ip = butil::endpoint2str(cntl->remote_side()).c_str();
    std::string server_ip = butil::endpoint2str(cntl->local_side()).c_str();
    std::string client_user;
    std::string req_body;
    brpc::HttpMethod http_method = header.method();
    std::string req_type = brpc::HttpMethod2Str(http_method);
    cntl->http_response().set_content_type("application/json");
    cntl->http_response().set_status_code(200);
    cntl->http_response().SetHeader("Access-Control-Allow-Origin", "*");
    cntl->http_response().SetHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, PUT, OPTIONS, HEAD");
    cntl->http_response().SetHeader("Access-Control-Allow-Headers", "Authorization, Origin, X-Requested-With, Content-Type, Accept");
    switch (http_method) {
    case brpc::HTTP_METHOD_POST:
    case brpc::HTTP_METHOD_PUT:
    case brpc::HTTP_METHOD_DELETE:
      cntl->request_attachment().copy_to(&req_body);
      break;
    case brpc::HTTP_METHOD_OPTIONS:
      return;
    default:
      break;
    }
    //int64_t queue_time = (int64_t)(butil::gettimeofday_us() - cntl->received_us()) / 1000;
    uint64_t time_start = mycommon::getMilliTime();
    std::string resp;
    std::string error;
    if ("v1/completions" == unresolved_path) {
      if (http_method != brpc::HTTP_METHOD_POST) {
        cntl->http_response().set_status_code(405);
        cntl->response_attachment().append(CompletionError("POST required", true));
        return;
      }
      rapidjson::Document doc;
      if (doc.Parse(req_body.data(), req_body.size()).HasParseError() || !doc.IsObject() ||
          !doc.HasMember("prompt") || !doc["prompt"].IsString() ||
          (doc.HasMember("max_tokens") && !doc["max_tokens"].IsInt()) ||
          (doc.HasMember("stream") && !doc["stream"].IsBool()) ||
          (doc.HasMember("ignore_eos") && !doc["ignore_eos"].IsBool()) ||
          (doc.HasMember("temperature") && (!doc["temperature"].IsNumber() || doc["temperature"].GetDouble() != 0)) ||
          (doc.HasMember("n") && (!doc["n"].IsInt() || doc["n"].GetInt() != 1))) {
        cntl->http_response().set_status_code(400);
        cntl->response_attachment().append(CompletionError("Expected a string prompt, integer max_tokens, boolean stream/ignore_eos; only temperature=0 and n=1 are supported", true));
        return;
      }
      Qwen3Engine::Request request;
      request.prompt = GET_JSON_STRING_BINARY(doc, "prompt", "");
      request.max_tokens = GET_JSON_INT(doc, "max_tokens", 128);
      request.ignore_eos = GET_JSON_BOOL(doc, "ignore_eos", false);
      const bool stream = GET_JSON_BOOL(doc, "stream", false);
      if (request.prompt.empty() || request.max_tokens <= 0 || request.max_tokens > FLAGS_max_seq_len) {
        cntl->http_response().set_status_code(400);
        cntl->response_attachment().append(CompletionError("Invalid prompt or max_tokens", true));
        return;
      }
      static std::atomic<uint64_t> sequence{0};
      const uint64_t created = time(NULL);
      const std::string id = "cmpl-" + std::to_string(created) + "-" + std::to_string(++sequence);
      butil::intrusive_ptr<brpc::ProgressiveAttachment> attachment;
      try {
        Qwen3Engine::TokenCallback on_token;
        if (stream) {
          on_token = [&](const std::string& text) {
            if (!attachment) {
              cntl->http_response().set_content_type("text/event-stream");
              cntl->http_response().SetHeader("Cache-Control", "no-cache");
              attachment = cntl->CreateProgressiveAttachment();
              done_guard.release()->Run();
              // Controller belongs to brpc after done; only use attachment.
            }
            const std::string chunk = "data: " + CompletionJson(id, created, text, nullptr) + "\n\n";
            return attachment->Write(chunk.data(), chunk.size()) == 0;
          };
        }
        auto result = _engine.generate(request, on_token);
        if (stream) {
          if (result.finish_reason != "cancelled") {
            if (!attachment) {  // EOS can arrive before any text is emitted.
              cntl->http_response().set_content_type("text/event-stream");
              cntl->http_response().SetHeader("Cache-Control", "no-cache");
              attachment = cntl->CreateProgressiveAttachment();
              done_guard.release()->Run();
            }
            const std::string last = "data: " + CompletionJson(id, created, "", &result) + "\n\ndata: [DONE]\n\n";
            attachment->Write(last.data(), last.size());
          }
        }
        else {
          cntl->response_attachment().append(CompletionJson(id, created, result.text, &result));
        }
      }
      catch (const std::exception& error) {
        const bool invalid = dynamic_cast<const std::invalid_argument*>(&error) != nullptr;
        const std::string body = CompletionError(error.what(), invalid);
        if (attachment) {
          const std::string chunk = "data: " + body + "\n\ndata: [DONE]\n\n";
          attachment->Write(chunk.data(), chunk.size());
        }
        else {
          cntl->http_response().set_status_code(invalid ? 400 : 500);
          cntl->response_attachment().append(body);
        }
      }
      return;
    }
    else if ("start_profile" == unresolved_path || "stop_profile" == unresolved_path) {
      if (http_method != brpc::HTTP_METHOD_POST) {
        cntl->http_response().set_status_code(405);
        cntl->response_attachment().append(GenRespBody(-1, "POST required"));
        return;
      }
      try {
        std::string directory = FLAGS_profile_dir;
        if ("start_profile" == unresolved_path) {
          _engine.start_profile();
        }
        else {
          directory = _engine.stop_profile();
        }
        cntl->response_attachment().append(GenRespBody(0, "", directory));
      }
      catch (const std::invalid_argument& error) {
        cntl->http_response().set_status_code(400);
        cntl->response_attachment().append(GenRespBody(-1, error.what()));
      }
      catch (const std::exception& error) {
        cntl->http_response().set_status_code(500);
        cntl->response_attachment().append(GenRespBody(-1, error.what()));
      }
      return;
    }
    else if ("upgrade" == unresolved_path) {
      if (brpc::HTTP_METHOD_POST == http_method) {
        cntl->response_attachment().append("ok\n");
        return;
      }
    }
    else if ("noupgrade" == unresolved_path) {
      if (brpc::HTTP_METHOD_POST == http_method) {
        cntl->response_attachment().append("ok\n");
        return;
      }
    }
    else if ("status.html" == unresolved_path) {
      if (brpc::HTTP_METHOD_GET == http_method) {
        cntl->response_attachment().append("ok\n");
        return;
      }
    }
    else if ("system" == unresolved_path) {
      if (brpc::HTTP_METHOD_GET == http_method) {
        rapidjson::Document doc;
        rapidjson::Document::AllocatorType& allocator_result = doc.GetAllocator();
        rapidjson::Value json_val_result(rapidjson::kObjectType);
        JSON_ADD_INT(json_val_result, status, "status", 0);
        JSON_ADD_STRING(json_val_result, version, "version", std::string(APP_VERSION));
        rapidjson::StringBuffer sb;
        rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
        json_val_result.Accept(writer);
        cntl->response_attachment().append(sb.GetString());
        return;
      }
    }
    cntl->http_response().set_status_code(400);
    cntl->response_attachment().append(GenRespBody(-1, "Not Implementation"));
    LOG(INFO) << "Req Url[" << uri_path << "] Body[" << req_body << "] Resp[" << resp << "]";
    return;
  }
private:
  Qwen3Engine& _engine;
};

int main(int argc, char *argv[]) {
  gflags::SetVersionString(APP_VERSION);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
    
  if (FLAGS_daemonize) {
    Daemonize();
  }
  
  FLAGS_log_dir = "logs/";
  if (!mycommon::file_exist(FLAGS_log_dir)) {
    std::string error;
    int ret = mycommon::mkdirs(FLAGS_log_dir, error);
    if (0 != ret) {
      LOG(ERROR) << "LOG Create Directory Failed, [" << error << "]";
      return ret;
    }
  }
  FLAGS_logbuflevel = -1;
  FLAGS_minloglevel = (FLAGS_debug ? google::INFO : google::FATAL);
  //google::EnableLogCleaner(FLAGS_log_days);
  std::string log_name = "service";
  google::InitGoogleLogging(log_name.c_str());
  FLAGS_stderrthreshold = google::FATAL;
  FLAGS_alsologtostderr = false;
  FLAGS_colorlogtostderr = true;
  LOG(INFO) << "Start... Version[" << APP_VERSION << "]";

  if (FLAGS_model_path.empty()) {
    LOG(ERROR) << "--model_path is required";
    return 1;
  }
  std::unique_ptr<Qwen3Engine> engine;
  try {
    engine = std::make_unique<Qwen3Engine>(FLAGS_model_path, FLAGS_device,
                                         FLAGS_max_seq_len, FLAGS_decode_graph, FLAGS_profile_dir);
  }
  catch (const std::exception& error) {
    LOG(ERROR) << "Engine startup failed: " << error.what();
    return 1;
  }
  brpc::Server server;

  MergeServiceImpl search_svc(*engine);
  // Add services into server. Notice the second parameter, because the
  // service is put on stack, we don't want server to delete it, otherwise
  // use brpc::SERVER_OWNS_SERVICE.
  if (server.AddService(&search_svc, brpc::SERVER_DOESNT_OWN_SERVICE, "/* => default_method") != 0) {
    LOG(ERROR) << "Fail to add search_svc";
    return -1;
  }

  std::string http_ip_port = mycommon::str_format("%s:%d", FLAGS_ipv6 ? "[::0]" : "0.0.0.0", FLAGS_http_port);

  // Start the server.
  brpc::ServerOptions options;
  //options.internal_port = FLAGS_http_monitor_port;
  //options.idle_timeout_sec = FLAGS_idle_timeout_s;
  //options.mutable_ssl_options()->default_cert.certificate = FLAGS_certificate;
  //options.mutable_ssl_options()->default_cert.private_key = FLAGS_private_key;
  //options.mutable_ssl_options()->ciphers = FLAGS_ciphers;
  if (server.Start(http_ip_port.c_str(), &options) != 0) {
    LOG(ERROR) << "Fail to start HttpServer";
    return -1;
  }

  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  server.RunUntilAskedToQuit();
  LOG(INFO) << "Shutdown";

  LOG(INFO) << "Exit...";
  return 0;
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
