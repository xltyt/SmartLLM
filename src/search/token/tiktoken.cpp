/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.08.15 16:05:10
 *
\****************************************************/

#include "tiktoken.h"
#include <glog/logging.h>

TikTokenEncoding::TikTokenEncoding(
  const std::string& name,
  const std::string& pat_str, 
  const CoreBPE::HashMap<std::vector<uint8_t>, size_t>& mergeable_ranks,
  const CoreBPE::StringMap<size_t>& special_tokens,
  const std::optional<size_t>& explicit_n_vocab
  ) {
  LOG(INFO) << "TikTokenEncoding::TikTokenEncoding Init";
  
  // Creates an Encoding object.

  // See openai_public.py for examples of how to construct an Encoding object.
  // Args:
  //     name: The name of the encoding. It should be clear from the name of the encoding
  //         what behaviour to expect, in particular, encodings with different special tokens
  //         should have different names.
  //     pat_str: A regex pattern string that is used to split the input text.
  //     mergeable_ranks: A dictionary mapping mergeable token bytes to their ranks. The ranks
  //         must correspond to merge priority.
  //     special_tokens: A dictionary mapping special token strings to their token values.
  //     explicit_n_vocab: The number of tokens in the vocabulary. If provided, it is checked
  //         that the number of mergeable tokens and special tokens is equal to this number.
  _name = name;
  _pat_str = pat_str;
  _mergeable_ranks = mergeable_ranks;
  _special_tokens = special_tokens;

  //self.max_token_value = max(
  //    max(mergeable_ranks.values()), max(special_tokens.values(), default=0)
  //)
  //if explicit_n_vocab:
  //    assert len(mergeable_ranks) + len(special_tokens) == explicit_n_vocab
  //    assert self.max_token_value == explicit_n_vocab - 1

  _core_bpe = new CoreBPE(mergeable_ranks, special_tokens, pat_str);
}

TikTokenEncoding::~TikTokenEncoding() {
  delete _core_bpe;
}

std::vector<size_t> TikTokenEncoding::encode(
  const std::string& text,
  const std::unordered_set<std::string>& allowed_special_p /*= std::unordered_set<std::string>()*/,
  const std::unordered_set<std::string>& disallowed_special_p /*= std::unordered_set<std::string>({"all"})*/
  ) {

  std::unordered_set<std::string> allowed_special = allowed_special_p;
  if (allowed_special.find("all") != allowed_special.end()) {
    allowed_special.clear();
    for (auto _ : _special_tokens) {
      allowed_special.insert(_.first);
    }
  }
  
  std::unordered_set<std::string> disallowed_special = disallowed_special_p;
  if (disallowed_special.find("all") != disallowed_special.end()) {
    disallowed_special.clear();
    for (auto _ : _special_tokens) {
      disallowed_special.insert(_.first);
    }
    for (auto _ : allowed_special) {
      auto it = disallowed_special.find(_);
      if (it != disallowed_special.end()) {
        disallowed_special.erase(it);
      }
    }
  }
  if (disallowed_special.size()) {
    std::string inner;
    for (auto _ : disallowed_special) {
      if (inner.size()) {
        inner += "|";
      }
      inner += regex_escape(_);
    }
    int error_number;
    PCRE2_SIZE error_offset;
    pcre2_code *re = pcre2_compile(
      (PCRE2_SPTR)inner.c_str(),
      PCRE2_ZERO_TERMINATED,
      0,
      &error_number,
      &error_offset,
      NULL
    );
    if (re == NULL) {
      throw std::invalid_argument("Invalid disallowed_special pattern");
    }
    pcre2_match_data *match_data = pcre2_match_data_create_from_pattern(re, NULL);
    if (match_data == NULL) {
      pcre2_code_free(re);
      throw std::invalid_argument("malloc disallowed_special failed");
    }
    PCRE2_SPTR subject = (PCRE2_SPTR)text.c_str();
    PCRE2_SIZE subject_length = (PCRE2_SIZE)text.size();
    int rc = pcre2_match(
      re,
      subject,
      subject_length,
      0,
      0,
      match_data,
      NULL
    );
    pcre2_match_data_free(match_data);
    pcre2_code_free(re);
    if (rc > 0) {
      throw std::invalid_argument(
        "Encountered text corresponding to disallowed special token {token!r}.\n" \
        "If you want this text to be encoded as a special token, " \
        "pass it to `allowed_special`, e.g. `allowed_special={{{token!r}, ...}}`.\n" \
        "If you want this text to be encoded as normal text, disable the check for this token " \
        "by passing `disallowed_special=(enc.special_tokens_set - {{{token!r}}})`.\n" \
        "To disable this check for all special tokens, pass `disallowed_special=()`.\n" \
        );
    }
  }
  
  return _core_bpe->encode(text, allowed_special);
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
