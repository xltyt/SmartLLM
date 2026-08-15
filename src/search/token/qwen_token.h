/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.08.15 16:02:15
 *
\****************************************************/

#ifndef _QWEN_TOKEN_H__
#define _QWEN_TOKEN_H__

#include "tiktoken.h"

class QwenToken {
public:
  QwenToken(const std::string& dir);
  virtual ~QwenToken();

public:
  std::vector<size_t> encode(
    const std::string& text
    );

protected:
  TikTokenEncoding *_encoding = NULL;
};

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
