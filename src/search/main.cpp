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
  MergeServiceImpl() {
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
    if ("infer" == unresolved_path) {
      if (brpc::HTTP_METHOD_POST == http_method) {
        return;
      }
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

  // Generally you only need one Server.
  brpc::Server server;

  MergeServiceImpl search_svc;
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
