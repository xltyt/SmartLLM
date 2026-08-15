/****************************************************\
 *
 * Copyright (C) 2012 Baidu.com, Inc. All Rights Reserved
 * Last modified: 2024.04.12 17:17:48
 *
\****************************************************/

#ifndef _SYSTEM_UTILS_H__
#define _SYSTEM_UTILS_H__

#include <string>

namespace mycommon {
  int mysystem(const char *cmd, char *output, int maxlen);
  int mysystem(const char *cmd, std::string& output);
  void mysystem_format(const char *format, ...);
  std::string get_current_dir();
}

#endif

/* vim: set expandtab nu smartindent ts=2 sw=2 sts=2: */
