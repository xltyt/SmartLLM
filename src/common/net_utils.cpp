/****************************************************\
 *
 * Copyright (C) 2017 All Rights Reserved
 * Last modified: 2025.05.07 17:44:01
 *
\****************************************************/
 
#include "net_utils.h"
#include <malloc.h>
#include <string.h>
#include <curl/curl.h>
#include <curl/easy.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netinet/if_ether.h>
#include <net/if_arp.h>
#include <netpacket/packet.h>
#include <net/if.h>
#include <net/ethernet.h>
#include <arpa/inet.h>
#include <sstream>
#include <fcntl.h>
#include <glog/logging.h>

#include "string_utils.h"
#include "common.h"

namespace mycommon {
  
  void net_init() {
    curl_global_init(CURL_GLOBAL_ALL);
  }
  
  extern "C" size_t _ReadFunction(void* ptr, size_t size, size_t nmemb, void* stream) {
    std::string* str = dynamic_cast<std::string*>((std::string*)stream);
    char* pData = (char*)ptr;
    str->append(pData, size * nmemb);
    return size * nmemb;
  }
  
  int net_crawl_url(const char *url, std::string& buf, const char *agent /*= NULL*/, const std::vector<std::string> *headers /*= NULL*/, std::map<std::string, std::string> *resp_headers /*= NULL*/, const char *proxy /*= NULL*/, int timeout_connect /*= 3000*/, int timeout_read /*= 3000*/, long* http_code /*= NULL*/) {
    if (NULL == url) {
      return -1;
    }

    CURL *curl = curl_easy_init();
    if (NULL == curl) {
      return -1;
    }

    struct curl_slist *slist_headers = NULL;
    if (NULL != headers) {
      for (std::vector<std::string>::const_iterator iter = headers->begin(); iter != headers->end(); iter++) {
        slist_headers = curl_slist_append(slist_headers, iter->c_str());
      }
    }
    std::string resp_headers_str;
    if (
         0 != curl_easy_setopt(curl, CURLOPT_URL, url)
      || 0 != curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1)
      || 0 != curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, _ReadFunction)
      || 0 != curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&buf)
      || 0 != curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, timeout_connect)
      || 0 != curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_read)
      || 0 != curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1)
      || 0 != curl_easy_setopt(curl, CURLOPT_FORBID_REUSE, 1)
      || 0 != curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_0)
      || 0 != curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L)
      || 0 != curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L)
      || 0 != curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L)
      || 0 != curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L)
      || (NULL != agent && 0 != curl_easy_setopt(curl, CURLOPT_USERAGENT, agent))
      || (NULL != slist_headers && 0 != curl_easy_setopt(curl, CURLOPT_HTTPHEADER, slist_headers))
      || (NULL != proxy && 0 != curl_easy_setopt(curl, CURLOPT_PROXY, proxy))
      || (NULL != resp_headers && 0 != curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, _ReadFunction))
      || (NULL != resp_headers && 0 != curl_easy_setopt(curl, CURLOPT_HEADERDATA, (void *)&resp_headers_str))
      ) {
      if (slist_headers) {
        curl_slist_free_all(slist_headers);
        slist_headers = NULL;
      }
      curl_easy_cleanup(curl);
      curl = NULL;
      return -2;
    }
    int ret = curl_easy_perform(curl);
    if (slist_headers) {
      curl_slist_free_all(slist_headers);
      slist_headers = NULL;
    }
    if (ret == CURLcode::CURLE_OK) {
      if (NULL != http_code) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, http_code);
      }
    }
    curl_easy_cleanup(curl);
    curl = NULL;
    if (NULL != resp_headers) {
      std::vector<std::string> lines;
      str_split(resp_headers_str, "\r\n", lines);
      for (std::vector<std::string>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        std::string::size_type pos = iter->find(": ");
        if (pos != std::string::npos) {
          std::string key = iter->substr(0, pos);
          std::string val = iter->substr(pos + 2);
          std::map<std::string, std::string>::iterator iter_header = resp_headers->find(key);
          if (iter_header == resp_headers->end()) {
            resp_headers->insert(std::map<std::string, std::string>::value_type(key, val));
          }
          else {
            iter_header->second = val;
          }
        }
      }
    }
    return ret;
  }

  int net_post_url(const std::string& url, const std::string& param, std::string& buf, const char *agent /*= NULL*/, const std::vector<std::string> *headers /*= NULL*/, std::map<std::string, std::string> *resp_headers /*= NULL*/, const char *proxy /*= NULL*/, int timeout_connect /*= 3000*/, int timeout_read /*= 3000*/, long* http_code /*= NULL*/, std::vector<std::string> *vec_result_buf /*= NULL*/) {
    if (0 == url.size()) { 
      return -1;
    }

    CURL *curl = curl_easy_init();
    if (NULL == curl) {
      return -1;
    }
    
    struct curl_slist *slist_headers = NULL;
    bool is_stream = false;
    if (NULL != headers) {
      for (std::vector<std::string>::const_iterator iter = headers->begin(); iter != headers->end(); iter++) {
        slist_headers = curl_slist_append(slist_headers, iter->c_str());
        if (iter->find("event-stream") != std::string::npos) {
          is_stream = true;
        }
      }
    }
    std::string tmp_buf;
    std::string resp_headers_str;
    if ( 0 != curl_easy_setopt(curl, CURLOPT_URL, url.c_str())
      || 0 != curl_easy_setopt(curl, CURLOPT_POSTFIELDS, param.data())
      || 0 != curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, param.size())
      || 0 != curl_easy_setopt(curl, CURLOPT_POST, 1)
      || 0 != curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, _ReadFunction)
      || 0 != curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&tmp_buf)
      || 0 != curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L)
      || 0 != curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, timeout_connect)
      || 0 != curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_read)
      || 0 != curl_easy_setopt(curl, CURLOPT_FORBID_REUSE, 1)
      || 0 != curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_0)
      || 0 != curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L)
      || 0 != curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L)
      || 0 != curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L)
      || 0 != curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L)
      || (NULL != agent && 0 != curl_easy_setopt(curl, CURLOPT_USERAGENT, agent))
      || (NULL != slist_headers && 0 != curl_easy_setopt(curl, CURLOPT_HTTPHEADER, slist_headers))
      || (NULL != proxy && 0 != curl_easy_setopt(curl, CURLOPT_PROXY, proxy))
      || (NULL != resp_headers && 0 != curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, _ReadFunction))
      || (NULL != resp_headers && 0 != curl_easy_setopt(curl, CURLOPT_HEADERDATA, (void *)&resp_headers_str))
      || (is_stream && 0 != curl_easy_setopt(curl, CURLOPT_HTTP_TRANSFER_DECODING, 1L))
    ) {
      if (slist_headers) {
        curl_slist_free_all(slist_headers);
        slist_headers = NULL;
      }
      curl_easy_cleanup(curl);
      curl = NULL;   
      return -2; 
    }
    int ret = curl_easy_perform(curl);
    if (slist_headers) {
      curl_slist_free_all(slist_headers);
      slist_headers = NULL;
    }
    if (ret == CURLcode::CURLE_OK) {
      if (NULL != http_code) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, http_code);
      }
    }
    curl_easy_cleanup(curl);
    curl = NULL;
    if (NULL != resp_headers) {
      std::vector<std::string> lines;
      str_split(resp_headers_str, "\r\n", lines);
      for (std::vector<std::string>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        std::string::size_type pos = iter->find(": ");
        if (pos != std::string::npos) {
          std::string key = iter->substr(0, pos);
          std::string val = iter->substr(pos + 2);
          std::map<std::string, std::string>::iterator iter_header = resp_headers->find(key);
          if (iter_header == resp_headers->end()) {
            resp_headers->insert(std::map<std::string, std::string>::value_type(key, val));
          }
          else {
            iter_header->second = val;
          }
        }
      }
    }
    if (is_stream) {
      if (vec_result_buf) {
        std::vector<std::string> vec_buf;
        str_split(tmp_buf, "\r\n", vec_buf);
        *vec_result_buf = vec_buf;
      }
    }
    else {
      buf = tmp_buf;
    }
    return ret;
  }
  
  int net_put_url(const std::string& url, const std::string& param, std::string& buf, const char *agent /*= NULL*/, const std::vector<std::string> *headers /*= NULL*/, std::map<std::string, std::string> *resp_headers /*= NULL*/, const char *proxy /*= NULL*/, int timeout_connect /*= 3000*/, int timeout_read /*= 3000*/, long* http_code /*= NULL*/) {
    if (0 == url.size()) { 
      return -1;
    }

    CURL *curl = curl_easy_init();
    if (NULL == curl) {
      return -1;
    }
    
    struct curl_slist *slist_headers = NULL;
    if (NULL != headers) {
      for (std::vector<std::string>::const_iterator iter = headers->begin(); iter != headers->end(); iter++) {
        slist_headers = curl_slist_append(slist_headers, iter->c_str());
      }
    }
    std::string resp_headers_str;
    if ( 0 != curl_easy_setopt(curl, CURLOPT_URL, url.c_str())
      || 0 != curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT")
      || 0 != curl_easy_setopt(curl, CURLOPT_POSTFIELDS, param.data())
      || 0 != curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, param.size())
      || 0 != curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, _ReadFunction)
      || 0 != curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&buf)
      || 0 != curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L)
      || 0 != curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, timeout_connect)
      || 0 != curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_read)
      || 0 != curl_easy_setopt(curl, CURLOPT_FORBID_REUSE, 1)
      || 0 != curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_0)
      || 0 != curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L)
      || 0 != curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L)
      || 0 != curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L)
      || 0 != curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L)
      || (NULL != agent && 0 != curl_easy_setopt(curl, CURLOPT_USERAGENT, agent))
      || (NULL != slist_headers && 0 != curl_easy_setopt(curl, CURLOPT_HTTPHEADER, slist_headers))
      || (NULL != proxy && 0 != curl_easy_setopt(curl, CURLOPT_PROXY, proxy))
      || (NULL != resp_headers && 0 != curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, _ReadFunction))
      || (NULL != resp_headers && 0 != curl_easy_setopt(curl, CURLOPT_HEADERDATA, (void *)&resp_headers_str))
    ) {
      if (slist_headers) {
        curl_slist_free_all(slist_headers);
        slist_headers = NULL;
      }
      curl_easy_cleanup(curl);
      curl = NULL;   
      return -2; 
    }
    int ret = curl_easy_perform(curl);
    if (slist_headers) {
      curl_slist_free_all(slist_headers);
      slist_headers = NULL;
    }
    if (ret == CURLcode::CURLE_OK) {
      if (NULL != http_code) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, http_code);
      }
    }
    curl_easy_cleanup(curl);
    curl = NULL;
    if (NULL != resp_headers) {
      std::vector<std::string> lines;
      str_split(resp_headers_str, "\r\n", lines);
      for (std::vector<std::string>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        std::string::size_type pos = iter->find(": ");
        if (pos != std::string::npos) {
          std::string key = iter->substr(0, pos);
          std::string val = iter->substr(pos + 2);
          std::map<std::string, std::string>::iterator iter_header = resp_headers->find(key);
          if (iter_header == resp_headers->end()) {
            resp_headers->insert(std::map<std::string, std::string>::value_type(key, val));
          }
          else {
            iter_header->second = val;
          }
        }
      }
    }
    return ret;
  }
  
  int net_del_url(const std::string& url, const std::string& param, std::string& buf, const char *agent /*= NULL*/, const std::vector<std::string> *headers /*= NULL*/, std::map<std::string, std::string> *resp_headers /*= NULL*/, const char *proxy /*= NULL*/, int timeout_connect /*= 3000*/, int timeout_read /*= 3000*/, long* http_code /*= NULL*/) {
    if (0 == url.size()) { 
      return -1;
    }

    CURL *curl = curl_easy_init();
    if (NULL == curl) {
      return -1;
    }

    struct curl_slist *slist_headers = NULL;
    if (NULL != headers) {
      for (std::vector<std::string>::const_iterator iter = headers->begin(); iter != headers->end(); iter++) {
        slist_headers = curl_slist_append(slist_headers, iter->c_str());
      }
    }
    std::string resp_headers_str;
    if (
         0 != curl_easy_setopt(curl, CURLOPT_URL, url.c_str())
      || 0 != curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1)
      || 0 != curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE")
      || 0 != curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, _ReadFunction)
      || 0 != curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&buf)
      || 0 != curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, timeout_connect)
      || 0 != curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_read)
      || 0 != curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1)
      || 0 != curl_easy_setopt(curl, CURLOPT_FORBID_REUSE, 1)
      || 0 != curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_0)
      || 0 != curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L)
      || 0 != curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L)
      || 0 != curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L)
      || 0 != curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L)
      || (NULL != agent && 0 != curl_easy_setopt(curl, CURLOPT_USERAGENT, agent))
      || (NULL != slist_headers && 0 != curl_easy_setopt(curl, CURLOPT_HTTPHEADER, slist_headers))
      || (NULL != proxy && 0 != curl_easy_setopt(curl, CURLOPT_PROXY, proxy))
      || (NULL != resp_headers && 0 != curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, _ReadFunction))
      || (NULL != resp_headers && 0 != curl_easy_setopt(curl, CURLOPT_HEADERDATA, (void *)&resp_headers_str))
      ) {
      if (slist_headers) {
        curl_slist_free_all(slist_headers);
        slist_headers = NULL;
      }
      curl_easy_cleanup(curl);
      curl = NULL;
      return -2;
    }
    if (param.size()) {
      if ( 0 != curl_easy_setopt(curl, CURLOPT_POSTFIELDS, param.data())
        || 0 != curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, param.size())
        ) {
          if (slist_headers) {
            curl_slist_free_all(slist_headers);
            slist_headers = NULL;
          }
          curl_easy_cleanup(curl);
          curl = NULL;
          return -3;
        }
    }
    int ret = curl_easy_perform(curl);
    if (slist_headers) {
      curl_slist_free_all(slist_headers);
      slist_headers = NULL;
    }
    if (ret == CURLcode::CURLE_OK) {
      if (NULL != http_code) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, http_code);
      }
    }
    curl_easy_cleanup(curl);
    curl = NULL;
    if (NULL != resp_headers) {
      std::vector<std::string> lines;
      str_split(resp_headers_str, "\r\n", lines);
      for (std::vector<std::string>::iterator iter = lines.begin(); iter != lines.end(); iter++) {
        std::string::size_type pos = iter->find(": ");
        if (pos != std::string::npos) {
          std::string key = iter->substr(0, pos);
          std::string val = iter->substr(pos + 2);
          std::map<std::string, std::string>::iterator iter_header = resp_headers->find(key);
          if (iter_header == resp_headers->end()) {
            resp_headers->insert(std::map<std::string, std::string>::value_type(key, val));
          }
          else {
            iter_header->second = val;
          }
        }
      }
    }
    return ret;
  }

  static size_t _ReadFileCallback(void *ptr, size_t size, size_t nmemb, void *stream) {
    size_t retcode = fread(ptr, size, nmemb, (FILE *)stream);
    return retcode;
  }
  
  int net_sftp_upload_file(const std::string& target_host, const std::string& target_path, const std::string& filepath, const std::string& user, const std::string& passwd, int timeout_s /*= 10*/) {
    if (0 == target_path.size() || 0 == filepath.size() || 0 == user.size() || 0 == passwd.size()) { 
      return -1;
    }
    std::stringstream sstream;
    sstream << "sftp://" << target_host << "/" << target_path;
    std::string url = sstream.str();

    struct stat file_info;
    if (stat(filepath.c_str(), &file_info)) {
      return -2;
    }
    curl_off_t fsize = (curl_off_t)file_info.st_size;
    FILE *file = fopen(filepath.c_str(), "rb");
    if (NULL == file) {
      return -3;
    }

    CURL *curl = curl_easy_init();
    if (NULL == curl) {
      fclose(file);
      return -1;
    }
    if ( 0 != curl_easy_setopt(curl, CURLOPT_URL, url.c_str())
      || 0 != curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L)
      || 0 != curl_easy_setopt(curl, CURLOPT_READFUNCTION, _ReadFileCallback)
      || 0 != curl_easy_setopt(curl, CURLOPT_READDATA, file)
      || 0 != curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, (curl_off_t)fsize)
      || 0 != curl_easy_setopt(curl, CURLOPT_FTP_RESPONSE_TIMEOUT, timeout_s)
      || 0 != curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, timeout_s * 1000)
      || 0 != curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_s * 1000)
      || 0 != curl_easy_setopt(curl, CURLOPT_USERPWD, (user + ":" + passwd).c_str())
      || 0 != curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L)
      || 0 != curl_easy_setopt(curl, CURLOPT_FORBID_REUSE, 1)
      || 0 != curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_0)
      || 0 != curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L)
      || 0 != curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L)
      || 0 != curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L)
      || 0 != curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L)
    ) {
      fclose(file);
      curl_easy_cleanup(curl);
      curl = NULL;   
      return -2; 
    }
    int ret = curl_easy_perform(curl);
    fclose(file);
    curl_easy_cleanup(curl);
    curl = NULL;
    return ret;
  }
  
  static size_t _WriteFileCallback(void *ptr, size_t size, size_t nmemb, void *stream) {
    size_t retcode = fwrite(ptr, size, nmemb, (FILE *)stream);
    return retcode;
  }
  
  int net_sftp_download_file(const std::string& target_host, const std::string& target_path, const std::string& filepath, const std::string& user, const std::string& passwd, int timeout_s /*= 3*/) {
    if (0 == target_path.size() || 0 == filepath.size() || 0 == user.size() || 0 == passwd.size()) { 
      return -1;
    }
    std::stringstream sstream;
    sstream << "sftp://" << target_host << "/" << target_path;
    std::string url = sstream.str();
    LOG(INFO) << "Url[" << url << "] Passwd[" + (user + ":" + passwd) + "]";

    FILE *file = fopen(filepath.c_str(), "wb");
    if (NULL == file) {
      return -3;
    }

    char errbuf[CURL_ERROR_SIZE];
    CURL *curl = curl_easy_init();
    if (NULL == curl) {
      fclose(file);
      return -1;
    }
    if ( 0 != curl_easy_setopt(curl, CURLOPT_URL, url.c_str())
      || 0 != curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, _WriteFileCallback)
      || 0 != curl_easy_setopt(curl, CURLOPT_WRITEDATA, file)
      || 0 != curl_easy_setopt(curl, CURLOPT_FTP_RESPONSE_TIMEOUT, timeout_s)
      || 0 != curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, timeout_s * 1000)
      || 0 != curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_s * 1000)
      || 0 != curl_easy_setopt(curl, CURLOPT_USERPWD, (user + ":" + passwd).c_str())
      || 0 != curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L)
      || 0 != curl_easy_setopt(curl, CURLOPT_FORBID_REUSE, 1)
      || 0 != curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_0)
      || 0 != curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L)
      || 0 != curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L)
      || 0 != curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L)
      || 0 != curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L)
      || 0 != curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errbuf)
    ) {
      fclose(file);
      curl_easy_cleanup(curl);
      curl = NULL;   
      return -2; 
    }
    int ret = curl_easy_perform(curl);
    fclose(file);
    curl_easy_cleanup(curl);
    curl = NULL;
    if (CURLE_OK != ret) {
      LOG(ERROR) << errbuf;
    }
    return ret;
  }
  
  int net_download_file(const std::string& url, const std::string& filepath, int timeout_connect /*= 3000*/, int timeout_read /*= 3000*/) {
    if (0 == filepath.size()) { 
      return -1;
    }
    FILE *file = fopen(filepath.c_str(), "wb");
    if (NULL == file) {
      return -3;
    }

    char errbuf[CURL_ERROR_SIZE];
    CURL *curl = curl_easy_init();
    if (NULL == curl) {
      fclose(file);
      return -1;
    }
    if ( 0 != curl_easy_setopt(curl, CURLOPT_URL, url.c_str())
      || 0 != curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, _WriteFileCallback)
      || 0 != curl_easy_setopt(curl, CURLOPT_WRITEDATA, file)
      || 0 != curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, timeout_connect)
      || 0 != curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_read)
      || 0 != curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L)
      || 0 != curl_easy_setopt(curl, CURLOPT_FORBID_REUSE, 1)
      || 0 != curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_0)
      || 0 != curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L)
      || 0 != curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L)
      || 0 != curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L)
      || 0 != curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L)
      || 0 != curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errbuf)
    ) {
      fclose(file);
      curl_easy_cleanup(curl);
      curl = NULL;   
      return -2; 
    }
    int ret = curl_easy_perform(curl);
    fclose(file);
    curl_easy_cleanup(curl);
    curl = NULL;
    if (CURLE_OK != ret) {
      LOG(ERROR) << errbuf;
    }
    return ret;
  }

  std::string get_ip_string(uint32_t ip) {
    char buf[256] = {0};
    (void) inet_ntop(AF_INET, &ip, buf, _countof(buf));
    return std::string(buf);
  }

  uint32_t parse_ip(const char *ip_str) {
    uint32_t ip = inet_addr(ip_str);
    return ip;
  }
}

/* vim: set expandtab nu smartindent ts=2 sw=2 sts=2: */
