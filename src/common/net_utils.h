/****************************************************\
 *
 * Copyright (C) 2017 All Rights Reserved
 * Last modified: 2025.05.07 17:44:01
 *
\****************************************************/


#ifndef _NET_UTILS_H__
#define _NET_UTILS_H__

#include <string>
#include <set>
#include <map>
#include <vector>
#include <string>
#include <stdint.h>

namespace mycommon {
  void net_init();
  int net_crawl_url(const char *url, std::string& buf, const char *agent = NULL, const std::vector<std::string> *headers = NULL, std::map<std::string, std::string> *resp_headers = NULL, const char *proxy = NULL, int timeout_connect = 3000, int timeout_read = 3000, long* http_code = NULL);
  int net_post_url(const std::string& url, const std::string& param, std::string& buf, const char *agent = NULL, const std::vector<std::string> *headers = NULL, std::map<std::string, std::string> *resp_headers = NULL, const char *proxy = NULL, int timeout_connect = 3000, int timeout_read = 3000, long* http_code = NULL, std::vector<std::string> *vec_result_buf = NULL);
  int net_put_url(const std::string& url, const std::string& param, std::string& buf, const char *agent = NULL, const std::vector<std::string> *headers = NULL, std::map<std::string, std::string> *resp_headers = NULL, const char *proxy = NULL, int timeout_connect = 3000, int timeout_read = 3000, long* http_code = NULL);
  int net_del_url(const std::string& url, const std::string& param, std::string& buf, const char *agent = NULL, const std::vector<std::string> *headers = NULL, std::map<std::string, std::string> *resp_headers = NULL, const char *proxy = NULL, int timeout_connect = 3000, int timeout_read = 3000, long* http_code = NULL);
  int net_sftp_upload_file(const std::string& target_host, const std::string& target_path, const std::string& filepath, const std::string& user, const std::string& passwd, int timeout_s = 3);
  int net_sftp_download_file(const std::string& target_host, const std::string& target_path, const std::string& filepath, const std::string& user, const std::string& passwd, int timeout_s = 3);
  int net_download_file(const std::string& url, const std::string& filepath, int timeout_connect = 3000, int timeout_read = 3000);
  std::string get_ip_string(uint32_t ip);
  uint32_t parse_ip(const char *ip_str);
}

#endif

/* vim: set expandtab nu smartindent ts=2 sw=2 sts=2: */
