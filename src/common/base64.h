#ifndef _COMMON_BASE_64_H
#define _COMMON_BASE_64_H

#include <string>

namespace common {
  std::string base64_encode(unsigned char const*, unsigned int len);
  std::string base64_encode(const std::string& s);
  std::string base64_decode(const std::string& s);
}

#endif

/* vim: set expandtab nu smartindent ts=2 sw=2 sts=2: */
