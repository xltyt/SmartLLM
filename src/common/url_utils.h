/****************************************************\
 *
 * Copyright (C) 2017 All Rights Reserved
 * Last modified: 2025.05.07 17:44:01
 *
\****************************************************/


#ifndef _URL_UTILS_H__
#define _URL_UTILS_H__

#include <string>

namespace mycommon {
  std::string uri_decode(const std::string& sSrc);
  std::string uri_encode(const std::string& sSrc);
  std::string url_encode_gbk(const std::string& URL);
  std::string url_decode_gbk(const std::string& URL);
}

#endif

/* vim: set expandtab nu smartindent ts=2 sw=2 sts=2: */
