#include "string_utils.h"
#include <boost/format.hpp>    
#include <boost/tokenizer.hpp>    
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <sstream>
#include <map>
#include <stdio.h>
#include <stdarg.h>
#include <inttypes.h>

#include <unicode/unistr.h>
#include <unicode/ustream.h>
#include <unicode/locid.h>
#include <unicode/normlzr.h>

namespace mycommon {
  void str_split(const std::string& str, const std::string& delim, std::vector<std::string>& tokens, bool compress /*= true*/) {
    boost::split(tokens, str, boost::is_any_of(delim), compress ? boost::algorithm::token_compress_on : boost::algorithm::token_compress_off);
  }
  void str_split(const std::string& str, const std::string& delim, std::set<std::string>& set_tokens, bool compress /*= true*/) {
    std::vector<std::string> tokens;
    boost::split(tokens, str, boost::is_any_of(delim), compress ? boost::algorithm::token_compress_on : boost::algorithm::token_compress_off);
    for (std::vector<std::string>::iterator iter = tokens.begin(); iter != tokens.end(); iter++) {
      if (0 == iter->size()) {
        continue;
      }
      set_tokens.insert(*iter);
    }
  }
  void str_split_int(const std::string& str, const std::string& delim, std::vector<int>& ids) {
    std::vector<std::string> tokens;
    boost::split(tokens, str, boost::is_any_of(delim));
    for (std::vector<std::string>::iterator iter = tokens.begin(); iter != tokens.end(); iter++) {
      if (0 == iter->size()) {
        continue;
      }
      int id = atoi(iter->c_str());
      ids.push_back(id);
    }
  }
  void str_split_int(const std::string& str, const std::string& delim, std::set<int>& ids) {
    std::vector<std::string> tokens;
    boost::split(tokens, str, boost::is_any_of(delim));
    for (std::vector<std::string>::iterator iter = tokens.begin(); iter != tokens.end(); iter++) {
      if (0 == iter->size()) {
        continue;
      }
      int id = atoi(iter->c_str());
      ids.insert(id);
    }
  }
  void str_split_float(const std::string& str, const std::string& delim, std::vector<float>& ids) {
    std::vector<std::string> tokens;
    boost::split(tokens, str, boost::is_any_of(delim));
    for (std::vector<std::string>::iterator iter = tokens.begin(); iter != tokens.end(); iter++) {
      if (0 == iter->size()) {
        continue;
      }
      float id = atof(iter->c_str());
      ids.push_back(id);
    }
  }
  void str_split_double(const std::string& str, const std::string& delim, std::vector<double>& ids) {
    std::vector<std::string> tokens;
    boost::split(tokens, str, boost::is_any_of(delim));
    for (std::vector<std::string>::iterator iter = tokens.begin(); iter != tokens.end(); iter++) {
      if (0 == iter->size()) {
        continue;
      }
      double id = atof(iter->c_str());
      ids.push_back(id);
    }
  }
  void str_replace(std::string& str, const std::string& src, const std::string& dst) {
    boost::replace_all(str, src, dst);
  }
  std::string str_replace_i(const std::string& str, const std::string& src, const std::string& dst) {
    std::string str_new = str;
    boost::replace_all(str_new, src, dst);
    return str_new;
  }
  void str_trim(std::string& str, const std::string& delim /*= "\r\n\t "*/, bool left /*= true*/, bool right /*= true*/) {
    if (left) {
      boost::trim_left_if(str, boost::is_any_of(delim));
    }
    if (right) {
      boost::trim_right_if(str, boost::is_any_of(delim));
    }
  }
  std::string str_join(const std::vector<std::string>& strs, const std::string& delim /*= ","*/) {
    std::string dst;
    for (int i = 0; i < (int)strs.size(); i++) {
      if (0 != dst.size()) {
        dst += delim;
      }
      dst += strs[i];
    }
    return dst;
  }
  
  std::string str_join(const std::set<std::string>& strs, const std::string& delim /*= ","*/) {
    std::string dst;
    for (std::set<std::string>::const_iterator iter = strs.begin(); iter != strs.end(); iter++) {
      if (0 != dst.size()) {
        dst += delim;
      }
      dst += *iter;
    }
    return dst;
  }

  std::string str_join(const std::set<int>& strs, const std::string& delim /*= ","*/) {
    std::string dst;
    for (std::set<int>::const_iterator iter = strs.begin(); iter != strs.end(); iter++) {
      int id = *iter;
      if (0 != dst.size()) {
        dst += delim;
      }
      char tmp[64] = {0};
      sprintf(tmp, "%d", id);
      dst += tmp;
    }
    return dst;
  }
  
  std::string str_from_int(int val) {
    char tmp[128];
    sprintf(tmp, "%d", val);
    return std::string(tmp);
  }
  
  std::string str_from_uint64(uint64_t val) {
    char tmp[128];
    sprintf(tmp, "%" PRIu64, val);
    return std::string(tmp);
  }
  
  bool str_startswith(const std::string& str, const std::string& src) {
    if (str.size() < src.size()) {
      return false;
    }
    std::string value = str.substr(0, src.size());
    if (value == src) {
      return true;
    }
    return false;
  }
  
  void split_multi(std::vector<std::string>& fields, const std::string& src, const std::string& separator) {
    size_t last  = 0;
    size_t index = 0;
    do{
      index = src.find_first_of(separator, last);
      if (index == std::string::npos) {
        index = src.size();
      }
      else {
        std::string value = src.substr(last, index - last);
        last = index + 1;
        fields.push_back(value);
      }
    }
    while(index != src.size());
  }

  std::map<std::string, std::map<std::string, std::string> > generate_custom(const std::string& custom_str) {
    std::map<std::string, std::map<std::string, std::string> > result_map;
    std::vector<std::string> fields;
    split_multi(fields, custom_str, "||");
    for(unsigned int i = 0; i < fields.size(); ++i) {
      std::vector<std::string> sub_fields;
      split_multi(sub_fields, fields[i], "::");
      if (2 != sub_fields.size()) {
        continue;
      }

      std::map<std::string, std::string> sub_map;
      std::vector<std::string> kvs;
      str_split(fields[1], ";", kvs);
      for(unsigned int j = 0; j < kvs.size(); ++j) {
        std::vector<std::string> kv;
        str_split(kvs[1], ":", kv);
        if (2 != kv.size()) {
          continue;
        }
        sub_map.insert(make_pair(kv[0], kv[1]));
      }
      if (sub_map.size() > 0) {
        result_map.insert(make_pair(fields[0], sub_map));
      }
    }
    return result_map;
  }

  void format_number(uint64_t num, std::string& str) {
    int digit[16] = {0};
    int i = 0;
    while (num) {
      digit[i++] = (int)(num % 1000);
      num /= 1000;
    }
    str = "";
    for (int j = i - 1; j >= 0; j--) {
      char tmp[64] = {0};
      if (j == i - 1) {
        sprintf(tmp, "%d", digit[j]);
      }
      else {
        str += ",";
        sprintf(tmp, "%03d", digit[j]);
      }
      str += tmp;
    }
  }
  
  bool is_int(const std::string& str) {
    bool ret = false;
    try {
      boost::lexical_cast<int>(str);
      ret = true;
    }
    catch (boost::bad_lexical_cast &) {
    }
    return ret;
  }
  
  bool is_uint64(const std::string& str) {
    bool ret = false;
    try {
      boost::lexical_cast<uint64_t>(str);
      ret = true;
    }
    catch (boost::bad_lexical_cast &) {
    }
    return ret;
  }

  bool is_float(const std::string& str) {
    bool ret = false;
    try {
      boost::lexical_cast<float>(str);
      ret = true;
    }
    catch (boost::bad_lexical_cast &) {
    }
    return ret;
  }
  
  std::string str_format(const char *format, ...) {
    char buf[8192] = {0};
    va_list args;
    va_start(args, format);
    vsnprintf(buf, 8192 - 1, format, args);
    va_end(args);
    return std::string(buf);
  }

  std::string str_lower_full(const std::string& text) {
    icu::UnicodeString query_unicode(text.c_str(), "UTF-8");
    std::string query;
    query_unicode.toLower().toUTF8String(query);
    return query;
  }

  int cacu_utf8_len(const std::string& text) {
    int utf8_len = 0;
    int len = text.size();
    int pos = 0;
    const char *p = text.c_str();
    for (pos = 0; pos < len; ) { 
      unsigned char c = *p; 
      int n = 0;
      if ((c & 0x80) == 0)        
        n = 1;
      else if ((c & 0xE0) == 0xC0) 
        n = 2;
      else if ((c & 0xF0) == 0xE0) 
        n = 3;
      else if ((c & 0xF8) == 0xF0) 
        n = 4;
      else if ((c & 0xFC) == 0xF8) 
        n = 5;
      else if ((c & 0xFE) == 0xFC) 
        n = 6;
      else 
        break;
      if (pos + n > len) {
        break;
      }
      p += n;
      pos += n;
      utf8_len++;
    }
    return utf8_len;
  }
  
  int cacu_utf8_len_raw(const std::string& text, int char_len) {
    int utf8_len = 0;
    int len = text.size();
    int pos = 0;
    const char *p = text.c_str();
    for (pos = 0; pos < len; ) { 
      unsigned char c = *p; 
      int n = 0;
      if ((c & 0x80) == 0)        
        n = 1;
      else if ((c & 0xE0) == 0xC0) 
        n = 2;
      else if ((c & 0xF0) == 0xE0) 
        n = 3;
      else if ((c & 0xF8) == 0xF0) 
        n = 4;
      else if ((c & 0xFC) == 0xF8) 
        n = 5;
      else if ((c & 0xFE) == 0xFC) 
        n = 6;
      else 
        break;
      if (pos + n > len) {
        break;
      }
      p += n;
      pos += n;
      utf8_len++;
      if (utf8_len >= char_len) {
        return pos;
      }
    }
    return len;
  }
  
  void str_unique_add(std::vector<std::string>& strs, const std::string& str) {
    for (std::vector<std::string>::const_iterator iter = strs.begin(); iter != strs.end(); iter++) {
      if (*iter == str) {
        return;
      }
    }
    strs.push_back(str);
  }
  
  void str_del(std::vector<std::string>& strs, const std::string& str) {
    for (std::vector<std::string>::iterator iter = strs.begin(); iter != strs.end();) {
      if (*iter == str) {
        strs.erase(iter++);
      }
      else {
        iter++;
      }
    }
  }
  
  std::string utf8_normalize(const std::string& input) {
    UNormalizationMode mode = UNORM_NFC;
    UErrorCode status = U_ZERO_ERROR;
    icu::UnicodeString src = icu::UnicodeString::fromUTF8(input);
    icu::UnicodeString dest;
    icu::Normalizer::normalize(src, mode, 0, dest, status);
    if (U_FAILURE(status)) return input;
    std::string result;
    dest.toUTF8String(result);
    return result;
  }
}

/* vim: set expandtab nu smartindent ts=2 sw=2 sts=2: */
