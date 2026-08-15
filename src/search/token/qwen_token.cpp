/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.08.13 23:11:34
 *
\****************************************************/

#include "qwen_token.h"
#include <iomanip>
#include <sstream>
#include <glog/logging.h>
#include <file_utils.h>
#include <common.h>

QwenToken::QwenToken(const std::string& dir) {
  std::string tokenizer_file = dir + "/tokenizer.json";
  auto ranks = CoreBPE::data_gym_to_mergeable_bpe_ranks(tokenizer_file);
  int n_vocab = ranks.size();

  std::vector<std::string> specials;
  CoreBPE::StringMap<size_t> special_tokens;
  {
		std::string json_content = mycommon::file_read(tokenizer_file);
    rapidjson::Document doc; 
    rapidjson::ParseResult ok = doc.Parse(json_content.c_str());
    if (ok && doc.IsObject()) {
      if (doc.HasMember("added_tokens") && doc["added_tokens"].IsArray()) {
        for (int j = 0; j < doc["added_tokens"].Size(); j++) {
          rapidjson::Value& json_add_token = doc["added_tokens"][j];
          if (json_add_token.IsObject()) {
            int token_id = GET_JSON_INT(json_add_token, "id", -1);
            std::string token_content = GET_JSON_STRING(json_add_token, "content", "");
            bool token_special = GET_JSON_BOOL(json_add_token, "special", false);
            if (-1 == token_id) {
              LOG(WARNING) << "data_gym_to_mergeable_bpe_ranks Load AddToken[" << j << "] Id Empty";
              continue;
            }
            if (!token_special) {
              continue;
            }
            special_tokens[token_content] = token_id;
            n_vocab++;
          }
        }
      }
    }
  }
  LOG(INFO) << "Special Len[" << special_tokens.size() << "] Vocab Count[" << n_vocab << "]";
  _encoding = new TikTokenEncoding(
    "Qwen",
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    ranks,
    special_tokens,
    n_vocab
    );
}

QwenToken::~QwenToken() {
  delete _encoding;
}

std::vector<size_t> QwenToken::encode(
  const std::string& text
  ) {
  return _encoding->encode(
	  text,
    std::unordered_set<std::string>({"all"}),
    std::unordered_set<std::string>({"all"})
    );
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
