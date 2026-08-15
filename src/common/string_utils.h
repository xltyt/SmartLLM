/****************************************************\
 *
 * Copyright (C) 2017 All Rights Reserved
 * Last modified: 2026.08.12 12:59:00
 *
\****************************************************/


#ifndef _STRING_UTILS_H__
#define _STRING_UTILS_H__

#include <string>
#include <vector>
#include <set>
#include <map>
#include <stdint.h>

namespace mycommon {
  void str_split(const std::string& str, const std::string& delim, std::vector<std::string>& tokens, bool compress = true);
  void str_split(const std::string& str, const std::string& delim, std::set<std::string>& tokens, bool compress = true);
  void str_split_int(const std::string& str, const std::string& delim, std::vector<int>& ids);
  void str_split_int(const std::string& str, const std::string& delim, std::set<int>& ids);
  void str_split_float(const std::string& str, const std::string& delim, std::vector<float>& ids);
  void str_split_double(const std::string& str, const std::string& delim, std::vector<double>& ids);
  void str_replace(std::string& str, const std::string& src, const std::string& dst);
  std::string str_replace_i(const std::string& str, const std::string& src, const std::string& dst);
  void str_trim(std::string& str, const std::string& delim = "\r\n\t ", bool left = true, bool right = true);
  std::string str_join(const std::vector<std::string>& strs, const std::string& delim = ",");
  template<class _T>
  std::string str_join(const std::vector<_T>& strs, const std::string& delim = ",") {
    std::string dst;
    for (int i = 0; i < (int)strs.size(); i++) {
      if (0 != dst.size()) {
        dst += delim;
      }
      dst += std::to_string(strs[i]);
    }
    return dst;
  }
  std::string str_join(const std::set<std::string>& strs, const std::string& delim = ",");
  std::string str_join(const std::set<int>& strs, const std::string& delim = ",");
  std::string str_from_int(int val);
  std::string str_from_uint64(uint64_t val);
  bool str_startswith(const std::string& str, const std::string& src);
  void split_multi(std::vector<std::string>& fields, const std::string& src, const std::string& separator);
  std::map<std::string, std::map<std::string, std::string> > generate_custom(const std::string& custom_str);
  void format_number(uint64_t num, std::string& str);
  bool is_int(const std::string& str);
  bool is_uint64(const std::string& str);
  bool is_float(const std::string& str);
  std::string str_format(const char *format, ...);
  std::string str_lower_full(const std::string& text);
  int cacu_utf8_len(const std::string& text);
  int cacu_utf8_len_raw(const std::string& text, int char_len);
  void str_unique_add(std::vector<std::string>& strs, const std::string& str);
  void str_del(std::vector<std::string>& strs, const std::string& str);
  std::string utf8_normalize(const std::string& input);
}

#endif

/* vim: set expandtab nu smartindent ts=2 sw=2 sts=2: */
