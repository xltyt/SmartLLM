/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2025.05.07 18:07:16
 *
\****************************************************/

#ifndef _TIKTOKEN_ENCODING_H__
#define _TIKTOKEN_ENCODING_H__

#include "core_bpe.h"

class TikTokenEncoding {
public:
  TikTokenEncoding(
    const std::string& name,
    const std::string& pat_str,
    const CoreBPE::HashMap<std::vector<uint8_t>, size_t>& mergeable_ranks,
    const CoreBPE::StringMap<size_t>& special_tokens,
    const std::optional<size_t>& explicit_n_vocab
    );
  virtual ~TikTokenEncoding();

public:
  std::vector<size_t> encode(
    const std::string& text,
    const std::unordered_set<std::string>& allowed_special = std::unordered_set<std::string>(),
    const std::unordered_set<std::string>& disallowed_special = std::unordered_set<std::string>({"all"})
    );

protected:
  std::string _name;
  std::string _pat_str;
  CoreBPE::HashMap<std::vector<uint8_t>, size_t> _mergeable_ranks;
  CoreBPE::StringMap<size_t> _special_tokens;
  std::optional<size_t> _explicit_n_vocab;
  CoreBPE *_core_bpe = NULL;
};

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
