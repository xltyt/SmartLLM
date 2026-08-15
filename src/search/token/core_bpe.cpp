/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.08.15 16:04:26
 *
\****************************************************/

#include "core_bpe.h"
#include <optional>
#include <thread>
#include <assert.h>
#include <glog/logging.h>
#include <unicode/uchar.h>
#include <boost/regex.hpp>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <string_utils.h>
#include <file_utils.h>
#include <base64.h>
#include <common.h>

std::string regex_escape(const std::string& text) {
  std::string result;
  result.reserve(text.size() * 2);
  for (char c : text) {
    switch (c) {
      case '\\': case '.': case '+': case '*': case '?':
      case '(': case ')': case '|': case '[': case ']':
      case '{': case '}': case '^': case '$': case '#':
      case '-': case '&': case '~':
        result.push_back('\\');
        result.push_back(c);
        break;
      default:
        result.push_back(c);
        break;
    }
  }
  return result;
}

int SlotAllocator::allocate() {
  for (int word = 0; word < NUM_WORDS; ++word) {
    uint64_t old = bitmap_[word].load(std::memory_order_relaxed);

    while (true) {
      // 这个 word 已经全部占用
      if (old == UINT64_MAX)
        break;

      // 找最低位的 0
      uint64_t free_bits = ~old;
      int bit = __builtin_ctzll(free_bits);

      uint64_t new_value = old | (1ULL << bit);

      // CAS：只有一个线程能成功
      if (bitmap_[word].compare_exchange_weak(
            old,
            new_value,
            std::memory_order_acquire,
            std::memory_order_relaxed)) {
        return word * 64 + bit;
      }

      // CAS 失败：
      // old 已经被更新成当前值，重新尝试
    }
  }

  return -1;
}

void SlotAllocator::release(int index) {
  int word = index / 64;
  int bit  = index % 64;

  bitmap_[word].fetch_and(
      ~(1ULL << bit),
      std::memory_order_release);
}

inline std::optional<size_t> get_rank(
  const std::vector<std::pair<size_t, size_t>>& parts,
  const CoreBPE::HashMap<std::string, size_t>& ranks,
  const std::string_view& piece,
  size_t start_idx,
  size_t skip
  ) {
  if ((start_idx + skip + 2) < parts.size()) {
    size_t begin = parts[start_idx].first;
    size_t end = parts[start_idx + skip + 2].first;
    //std::string_view sub = piece.substr(begin, end - begin);
    std::string_view sub(piece.data() + begin, end - begin);
    auto it = ranks.find(sub);
    if (it != ranks.end()) {
      return it->second;
    }
  }
  return std::nullopt;
}

template <typename F>
std::vector<typename std::result_of<F(std::pair<size_t, size_t>)>::type>
_byte_pair_merge(
  const std::string_view& piece,
  const CoreBPE::HashMap<std::string, size_t>& ranks,
  F f
  ) {
  using T = typename std::result_of<F(std::pair<size_t, size_t>)>::type;
  
  // This is a vector of (start, rank).
  // The rank is of the byte pair starting at position start.
  // The rank of the last item in the vector is not a valid value.
  std::vector<std::pair<size_t, size_t>> parts;
  parts.reserve(piece.size() + 1);
  for (size_t i = 0; i <= piece.size(); ++i) {
    parts.emplace_back(i, std::numeric_limits<size_t>::max());
  }
  
  // We look up the ranks once in the beginning and iteratively update
  // them during each merge, which reduces the number of rank lookups.
  for (size_t i = 0; i + 2 < parts.size(); ++i) {
    auto r = get_rank(parts, ranks, piece, i, 0);
    if (r.has_value()) {
      parts[i].second = *r;
    }
  }
  
  // If you have n parts and m merges, this does O(mn) work.
  // We could do something with a heap and do O(m log n) work.
  // It is important to consider that n is often small (<100), and as such
  // the cache-locality benefits outweigh the algorithmic complexity downsides
  // of the `parts` vector data structure above.

  // Note that we hash bytes, not token pairs. As long as we train BPE the way we
  // currently do, this is equivalent. An easy way to break this would be to decouple
  // merge priority from token index or to prevent specific token merges.
    
  while (parts.size() > 1) {
    // usize::MAX is a sentinel rank value allowing us to
    // take the min more quickly
    size_t min_rank = std::numeric_limits<size_t>::max();
    size_t min_rank_idx = 0;
    for (size_t i = 0; i < parts.size() - 1; ++i) {
      if (parts[i].second < min_rank) {
        min_rank = parts[i].second;
        min_rank_idx = i;
      }
    }

    if (min_rank != std::numeric_limits<size_t>::max()) {
      
      // NOTE: We are about to remove parts[i + 1]. We do not do it
      // yet because there are cache-locality benefits to updating
      // parts[i] and parts[i-1] before removing, which could thrash
      // the cache. Thus, we update the rank calculation by skipping over
      // parts[i + 1], by invoking `get_rank!` with `skip = 1`.
            
      parts[min_rank_idx].second = get_rank(parts, ranks, piece, min_rank_idx, 1).value_or(std::numeric_limits<size_t>::max());

      if (min_rank_idx > 0) {
        parts[min_rank_idx - 1].second = get_rank(parts, ranks, piece, min_rank_idx - 1, 1).value_or(std::numeric_limits<size_t>::max());
      }
      parts.erase(parts.begin() + min_rank_idx + 1);
    }
    else {
      break;
    }
  }
  
  std::vector<T> out;
  out.reserve(parts.size() - 1);
  for (size_t i = 0; i + 1 < parts.size(); ++i) {
    out.push_back(f({parts[i].first, parts[i + 1].first}));
  }
  return out;
}

inline std::vector<size_t> byte_pair_encode(
  const std::string_view& piece,
  const CoreBPE::HashMap<std::string, size_t>& ranks
  ) {
  if (piece.size() == 1) {
    auto it = ranks.find(piece);
    if (it != ranks.end()) {
      return {it->second};
    }
    return {};
  }
  return _byte_pair_merge(piece, ranks, [&](std::pair<size_t, size_t> range) -> size_t {
		//std::string_view sub = piece.substr(range.first, range.second - range.first);
    std::string_view sub(piece.data() + range.first, range.second - range.first);
		auto it = ranks.find(sub);
    if (it != ranks.end()) {
      return it->second;
    }
    //__builtin_unreachable(); 
    throw std::runtime_error("byte_pair_encode: piece not in ranks");
  });
}

// Various performance notes:
//
// Regex
// =====
// Most of the time is spent in regex. The easiest way to speed this up is by using less fancy
// regex features. For instance, using a regex parse-able by `regex` crate is 3x faster than
// the usual regex we use.
//
// However, given that we're using a regex parse-able by `regex`, there isn't much difference
// between using the `regex` crate and using the `fancy_regex` crate.
//
// There is an important interaction between threading, `regex` and `fancy_regex`.
// When using `fancy_regex`, we hit `regex.find_at`. It turns out that this causes contention on
// some mutable scratch space inside of `regex`. This absolutely kills performance. When using plain
// old `regex`, we don't hit this, because `find_iter` has a different code path.
// Related: https://github.com/rust-lang/regex/blob/master/PERFORMANCE.md
// Anyway, the way we get around this is with having a (mostly) thread local clone of the regex for
// each thread.
//
// Threading
// =========
// I tried using `rayon`. It wasn't really faster than using Python threads and releasing the GIL.
// So goodbye `rayon`! Let thread count etc be in control of our Python users.
//
// Caching
// =======
// The reference tokeniser has an lru cache over the equivalent of `byte_pair_encode`.
// Originally, we had one too! Without it, we were only vaguely faster than Python.
// I used an RWLock to protect the cache. This didn't seem to hurt single threaded performance
// noticeably, but it did affect multi-threaded performance. Weirdly, it seemed to affect
// multi-threaded performance even when I only had readers (maybed I messed something up?).
// Anyway, I realised that we could get rid of the cache, if we treat the set of tokens as a cache!
// These are exactly the set or merges that are likely to be hot. And now we don't have to think
// about interior mutability, memory use, or cloning.
//
// Hashing
// =======
// We use FxHashMap instead of the standard HashMap. This is maybe like a 5-10% win?
// The current implementation ends up doing a lot of hashing of bytes. In theory, this could be made
// to be hashing of two-tuples of ints, which looks like it may also be a couple percent faster.
  
std::size_t CoreBPE::VectorHash::operator()(const std::vector<uint8_t>& v) const noexcept {
  std::size_t seed = v.size();
  for (auto& b : v) {
    seed ^= std::hash<uint8_t>{}(b) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

std::size_t CoreBPE::VectorHash::operator()(size_t k) const noexcept {
  return k;
}
  
bool CoreBPE::VectorEqual::operator()(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) const noexcept {
  return a == b;
}

bool CoreBPE::VectorEqual::operator()(size_t a, size_t b) const noexcept {
  return a == b;
}
  
std::size_t CoreBPE::VectorSizeTHash::operator()(const std::vector<size_t>& v) const noexcept {
  std::size_t seed = v.size();
  for (auto& x : v) {
    seed ^= std::hash<size_t>{}(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}
  
std::vector<uint32_t> CoreBPE::decode_utf8(const std::string& text) {
  std::vector<uint32_t> ret;
  int len = text.size();
  int pos = 0;
  const char *p = text.c_str();
  for (pos = 0; pos < len; ) { 
    unsigned char c = *p; 
    int n = 0;
    if ((c & 0x80) == 0) {
      n = 1;
      ret.push_back(c);
    }
    else if ((c & 0xE0) == 0xC0) {
      n = 2;
      if (pos + n > len) {
        break;
      }
      if ((text[pos + 1] & 0xC0) == 0x80) {
        uint32_t val =
          (uint32_t)(text[pos] & 0x3F) << 6 |
          (uint32_t)(text[pos + 1] & 0x3F);
        ret.push_back(val);
      }
      else {
        break;
      }
    }
    else if ((c & 0xF0) == 0xE0) {
      n = 3;
      if (pos + n > len) {
        break;
      }
      if ((text[pos + 1] & 0xC0) == 0x80 &&
          (text[pos + 2] & 0xC0) == 0x80) {
        uint32_t val = 
          (uint32_t)(text[pos] & 0x0F) << 12 |
          (uint32_t)(text[pos + 1] & 0x3F) << 6 |
          (uint32_t)(text[pos + 2] & 0x3F);
        ret.push_back(val);
      }
      else {
        break;
      }
    }
    else if ((c & 0xF8) == 0xF0) {
      n = 4;
      if (pos + n > len) {
        break;
      }
      if ((text[pos + 1] & 0xC0) == 0x80 &&
          (text[pos + 2] & 0xC0) == 0x80 &&
          (text[pos + 3] & 0xC0) == 0x80) {
        uint32_t val = 
          (uint32_t)(text[pos] & 0x07) << 18 |
          (uint32_t)(text[pos + 1] & 0x3F) << 12 |
          (uint32_t)(text[pos + 2] & 0x3F) << 6 |
          (uint32_t)(text[pos + 3] & 0x3F);
        ret.push_back(val);
      }
      else {
        break;
      }
    }
    else 
      break;
    if (pos + n > len) {
      break;
    }
    p += n;
    pos += n;
  }
  return ret;
}

CoreBPE::HashMap<std::vector<uint8_t>, size_t> CoreBPE::data_gym_to_mergeable_bpe_ranks(
  const std::string& vocab_bpe_file,
  const std::string& encoder_json_file
  ) {
#if 0
	std::u32string py_chr(int code_point) {
    if (code_point < 0 || code_point > 0x10FFFF) {
        throw std::out_of_range("Invalid Unicode code point");
    }
    // 检查代理区 (Surrogate range) 55296-57343 (0xD800-0xDFFF) 是无效的单独码点
    if (code_point >= 0xD800 && code_point <= 0xDFFF) {
        throw std::out_of_range("Code point in surrogate range");
    }
    return std::u32string(1, static_cast<char32_t>(code_point));
	}
#endif
	// U_IS_CODE_POINT

  // NB: do not add caching to this function
  std::vector<uint8_t> rank_to_intbyte;
  for (int b = 0; b < 256; b++) {
    if (U_IS_CODE_POINT(b) && u_isprint(b) && u_isgraph(b) && b != ' ') {
      rank_to_intbyte.push_back(static_cast<uint8_t>(b));
    }
  }
  std::map<uint32_t, uint8_t> data_gym_byte_to_byte;
  for (uint8_t b : rank_to_intbyte) {
    data_gym_byte_to_byte[static_cast<uint32_t>(b)] = b;
  }
  int n = 0;
  for (int b = 0; b < 256; b++) {
    if (std::find(rank_to_intbyte.begin(), rank_to_intbyte.end(), b) == rank_to_intbyte.end()) {
      rank_to_intbyte.push_back(static_cast<uint8_t>(b));
      data_gym_byte_to_byte[static_cast<uint32_t>(256 + n)] = static_cast<uint8_t>(b);
      n++;
    }
  }
  assert(rank_to_intbyte.size() == 256);
  //for (int i = 0; i < rank_to_intbyte.size(); i++) {
  //  LOG(INFO) << (int)rank_to_intbyte[i];
  //}
	//for (auto _ : data_gym_byte_to_byte) {
  //  LOG(INFO) << (int)_.first << "\t" << (int)_.second;
  //}
    
  // vocab_bpe contains the merges along with associated ranks
  std::vector<std::pair<std::string, std::string>> bpe_merges;
  std::vector<std::string> lines;
  mycommon::file_read(vocab_bpe_file, lines);
  for (int i = 1; i < lines.size(); i++) {
    std::vector<std::string> fields;
    mycommon::str_split(lines[i], " ", fields);
    if (2 == fields.size()) {
      //mycommon::str_trim(fields[0], "\r\n\t \xa0");
      //mycommon::str_trim(fields[1], "\r\n\t \xa0");
      bpe_merges.emplace_back(fields[0], fields[1]);
    }
  }
  LOG(INFO) << "BPE Merge Len[" << bpe_merges.size() << "]";
  //for (const auto& merge : bpe_merges) {
  //  LOG(INFO) << merge.first << "\t" << merge.second;
  //}
  
  auto ToString = [](const std::vector<uint8_t>& value) -> std::string {
    std::string tmp;
    for (auto _ : value) {
      if (tmp.size()) {
        tmp += " ";
      }
      tmp += std::to_string((int)_);
    }
    return tmp;
  };
  
  auto decode_data_gym = [&](const std::string& value) -> std::vector<uint8_t> {
    //LOG(INFO) << value << "," << value.size();
    std::vector<uint8_t> decoded;
    std::string tmp;
    for (uint32_t c : decode_utf8(value)) {
      //LOG(INFO) << (int)c;
      decoded.push_back(data_gym_byte_to_byte[c]);
    }
    //LOG(INFO) << ToString(decoded);
    //LOG(INFO) << decoded.size();
    return decoded;
  };
    
  // add the single byte tokens
  CoreBPE::HashMap<std::vector<uint8_t>, size_t> bpe_ranks;
  for (size_t i = 0; i < rank_to_intbyte.size(); i++) {
    bpe_ranks[{rank_to_intbyte[i]}] = static_cast<size_t>(i);
  }
    
  // add the merged tokens
  n = bpe_ranks.size();
  for (const auto& merge : bpe_merges) {
    std::vector<uint8_t> first_decoded = decode_data_gym(merge.first);
    std::vector<uint8_t> second_decoded = decode_data_gym(merge.second);
    first_decoded.insert(first_decoded.end(), second_decoded.begin(), second_decoded.end());
    bpe_ranks[first_decoded] = n;
    n++;
  }
  LOG(INFO) << "BPE Len[" << bpe_ranks.size() << "]";
    
  // check that the encoder file matches the merges file
  // this sanity check is important since tiktoken assumes that ranks are ordered the same
  // as merge priority
    
  CoreBPE::HashMap<std::vector<uint8_t>, size_t> encoder_json_loaded;
  {
		std::string json_content = mycommon::file_read(encoder_json_file);
    rapidjson::Document doc; 
    rapidjson::ParseResult ok = doc.Parse(json_content.c_str());
    if (ok && doc.IsObject()) {
      for (rapidjson::Value::MemberIterator iter_member = doc.MemberBegin(); iter_member != doc.MemberEnd(); ++iter_member) {
        rapidjson::Value& json_key = iter_member->name;
        rapidjson::Value& json_val = iter_member->value;
        if (!json_key.IsString() || !json_val.IsInt()) {
          continue;
        }    
        std::string key = json_key.GetString();
        int val = json_val.GetInt();
        encoder_json_loaded[decode_data_gym(key)] = val;
      }
    }
    else {
      size_t errLine = 1; 
      for (size_t i = 0; i < doc.GetErrorOffset() && i < json_content.size(); ++i) {
        if (json_content[i] == '\n') ++errLine;
      }    
      LOG(WARNING) << "Parse encoder.json Failed[" << rapidjson::GetParseError_En(doc.GetParseError()) << "] Offset[" << doc.GetErrorOffset() << "] Line[" << errLine << "]";
      throw std::runtime_error("parse encoder.json failed");
    }
    encoder_json_loaded.erase(decode_data_gym("<|endoftext|>"));
    encoder_json_loaded.erase(decode_data_gym("<|startoftext|>"));
  }
  LOG(INFO) << "Encoder Len[" << encoder_json_loaded.size() << "]";

  if (bpe_ranks != encoder_json_loaded) {
    throw std::runtime_error("BPE ranks do not match encoder.json");
  }
  
  return bpe_ranks;
}
  
CoreBPE::HashMap<std::vector<uint8_t>, size_t> CoreBPE::data_gym_to_mergeable_bpe_ranks(
  const std::string& tokenizer_file
  ) {
  
  // NB: do not add caching to this function
  std::vector<uint8_t> rank_to_intbyte;
  for (int b = 0; b < 256; b++) {
    if (U_IS_CODE_POINT(b) && u_isprint(b) && u_isgraph(b) && b != ' ') {
      rank_to_intbyte.push_back(static_cast<uint8_t>(b));
    }
  }
  std::map<uint32_t, uint8_t> data_gym_byte_to_byte;
  for (uint8_t b : rank_to_intbyte) {
    data_gym_byte_to_byte[static_cast<uint32_t>(b)] = b;
  }
  int n = 0;
  for (int b = 0; b < 256; b++) {
    if (std::find(rank_to_intbyte.begin(), rank_to_intbyte.end(), b) == rank_to_intbyte.end()) {
      rank_to_intbyte.push_back(static_cast<uint8_t>(b));
      data_gym_byte_to_byte[static_cast<uint32_t>(256 + n)] = static_cast<uint8_t>(b);
      n++;
    }
  }
  assert(rank_to_intbyte.size() == 256);

  auto ToString = [](const std::vector<uint8_t>& value) -> std::string {
    std::string tmp;
    for (auto _ : value) {
      if (tmp.size()) {
        tmp += " ";
      }
      tmp += std::to_string((int)_);
    }
    return tmp;
  };
  
  auto decode_data_gym = [&](const std::string& value) -> std::vector<uint8_t> {
    //LOG(INFO) << value << "," << value.size();
    std::vector<uint8_t> decoded;
    std::string tmp;
    for (uint32_t c : decode_utf8(value)) {
      //LOG(INFO) << (int)c;
      decoded.push_back(data_gym_byte_to_byte[c]);
    }
    //LOG(INFO) << ToString(decoded);
    //LOG(INFO) << decoded.size();
    return decoded;
  };
  
  std::string pattern;
  std::vector<std::pair<std::string, std::string>> bpe_merges;
  CoreBPE::HashMap<std::vector<uint8_t>, size_t> encoder_json_loaded;
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
          }
        }
      }
      if (doc.HasMember("pre_tokenizer") && doc["pre_tokenizer"].IsObject()) {
        rapidjson::Value& json_pre_token = doc["pre_tokenizer"];
        if (json_pre_token.HasMember("pretokenizers") && json_pre_token["pretokenizers"].IsArray()) {
          for (int j = 0; j < json_pre_token["pretokenizers"].Size(); j++) {
            rapidjson::Value& json_pretokenizer = json_pre_token["pretokenizers"][j];
            if (json_pretokenizer.IsObject()) {
              std::string type = GET_JSON_STRING(json_pretokenizer, "type", "");
              if ("Split" != type) {
                continue;
              }
              if (json_pretokenizer.HasMember("pattern") && json_pretokenizer["pattern"].IsObject()) {
                rapidjson::Value& json_pattern = json_pretokenizer["pattern"];
                pattern = GET_JSON_STRING(json_pattern, "Regex", "");
              }
            }
          }
        }
      }
      if (doc.HasMember("model") && doc["model"].IsObject()) {
        rapidjson::Value& json_model = doc["model"];
        if (json_model.HasMember("vocab") && json_model["vocab"].IsObject()) {
          rapidjson::Value& json_vocab = json_model["vocab"];
          for (rapidjson::Value::MemberIterator iter_member = json_vocab.MemberBegin(); iter_member != json_vocab.MemberEnd(); ++iter_member) {
            rapidjson::Value& json_key = iter_member->name;
            rapidjson::Value& json_val = iter_member->value;
            if (!json_key.IsString() || !json_val.IsInt()) {
              continue;
            }    
            std::string key = json_key.GetString();
            int val = json_val.GetInt();
            encoder_json_loaded[decode_data_gym(key)] = val;
          }
        }
        if (json_model.HasMember("merges") && json_model["merges"].IsArray()) {
          for (int j = 0; j < json_model["merges"].Size(); j++) {
            rapidjson::Value& json_merge = json_model["merges"][j];
            if (json_merge.IsArray() &&
                json_merge.Size() == 2 &&
                json_merge[0].IsString() &&
                json_merge[1].IsString()) {
              bpe_merges.emplace_back(json_merge[0].GetString(), json_merge[1].GetString());
            }
          }
        }
      }
    }
  }
  if (pattern.empty()) {
    throw std::runtime_error("Pattern Empty");
  }
  if (bpe_merges.empty()) {
    throw std::runtime_error("Merges Empty");
  }
  if (encoder_json_loaded.empty()) {
    throw std::runtime_error("Vocab Empty");
  }
  LOG(INFO) << "BPE Merge Len[" << bpe_merges.size() << "]";
    
  // add the single byte tokens
  CoreBPE::HashMap<std::vector<uint8_t>, size_t> bpe_ranks;
  for (size_t i = 0; i < rank_to_intbyte.size(); i++) {
    bpe_ranks[{rank_to_intbyte[i]}] = static_cast<size_t>(i);
  }
    
  // add the merged tokens
  n = bpe_ranks.size();
  for (const auto& merge : bpe_merges) {
    std::vector<uint8_t> first_decoded = decode_data_gym(merge.first);
    std::vector<uint8_t> second_decoded = decode_data_gym(merge.second);
    first_decoded.insert(first_decoded.end(), second_decoded.begin(), second_decoded.end());
    bpe_ranks[first_decoded] = n;
    n++;
  }
  LOG(INFO) << "BPE Len[" << bpe_ranks.size() << "]";
  
  LOG(INFO) << "Encoder Len[" << encoder_json_loaded.size() << "]";

  if (bpe_ranks != encoder_json_loaded) {
    throw std::runtime_error("BPE ranks do not match encoder.json");
  }
  return bpe_ranks;
}

#if 0
CoreBPE::HashMap<std::vector<uint8_t>, size_t> CoreBPE::load_tiktoken_bpe(const std::string& tiktoken_bpe_file) {
  CoreBPE::HashMap<std::vector<uint8_t>, size_t> bpe_ranks;
  // NB: do not add caching to this function
  std::vector<std::string> lines;
  mycommon::file_read(tiktoken_bpe_file, lines);
  for (int i = 0; i < lines.size(); i++) {
    std::vector<std::string> fields;
    mycommon::str_split(lines[i], " ", fields);
    if (2 == fields.size()) {
      std::string key = common::base64_decode(fields[0]);
      std::vector<uint8_t> keys;
      for (int j = 0; j < key.size(); j++) {
        keys.push_back((uint8_t)key[j]);
      }
      bpe_ranks[keys] = atoi(fields[1].c_str());
    }
  }
  return bpe_ranks;
}
#endif

CoreBPE::CoreBPE(
  const HashMap<std::vector<uint8_t>, size_t>& encoder,
  const StringMap<size_t>& special_tokens_encoder,
  const std::string& pattern
  ) {
  for (auto _ : encoder) {
    _encoder[std::string(_.first.begin(), _.first.end())] = _.second;
  }
  _encoder.max_load_factor(0.7);
  _special_tokens_encoder = special_tokens_encoder;
  
  // Regex
  pcre2_compile_context* compile_context = pcre2_compile_context_create(NULL);
  LOG(INFO) << "CoreBPE::CoreBPE Init Regex";
  {
    int pcre2_error_number;
    PCRE2_SIZE pcre2_error_offset;
    PCRE2_SPTR pattern_ptr = (PCRE2_SPTR)pattern.c_str();
    PCRE2_SIZE pattern_len = pattern.length();
    pcre2_code* regex = pcre2_compile(
      pattern_ptr,
      pattern_len,
      PCRE2_UTF | PCRE2_UCP | PCRE2_NO_UTF_CHECK,  // Enable UTF-8 & Unicode
      &pcre2_error_number,
      &pcre2_error_offset,
      compile_context
      );
    if (regex == NULL) {
      pcre2_code_free(regex);
      PCRE2_UCHAR buffer;
      pcre2_get_error_message(pcre2_error_number, &buffer, sizeof(buffer));
      throw std::invalid_argument("Invalid regex pattern[" + pattern + "] Error[" + std::string((char*)&buffer) + "]");
      //throw std::invalid_argument("Invalid regex pattern[" + pattern + "]");
    }
    _regex_tls.resize(MAX_NUM_THREADS, nullptr);
    _match_data_tls.resize(MAX_NUM_THREADS);
    for (size_t i = 0; i < MAX_NUM_THREADS; ++i) {
      _regex_tls[i] = pcre2_code_copy(regex);
      pcre2_jit_compile(_regex_tls[i], PCRE2_JIT_COMPLETE);
      _match_data_tls[i] = pcre2_match_data_create_from_pattern(_regex_tls[i], NULL);
    }
    pcre2_code_free(regex);
  }
  
  // Special Regex
  LOG(INFO) << "CoreBPE::CoreBPE Init Special Regex";
  std::string special_pattern;
  for (const auto& kv : _special_tokens_encoder) {
    if (!special_pattern.empty()) {
      special_pattern += "|";
    }
    special_pattern += regex_escape(kv.first);
  }
  if (!special_pattern.empty()) {
    int pcre2_error_number;
    PCRE2_SIZE pcre2_error_offset;
    PCRE2_SPTR special_pattern_ptr = (PCRE2_SPTR)special_pattern.c_str();
    PCRE2_SIZE special_pattern_len = special_pattern.length();
    pcre2_code* special_regex = pcre2_compile(
        special_pattern_ptr,
        special_pattern_len,
        PCRE2_UTF | PCRE2_UCP | PCRE2_NO_UTF_CHECK,
        &pcre2_error_number,
        &pcre2_error_offset,
        compile_context
        );
    if (special_regex == NULL) {
      pcre2_code_free(special_regex);
      //PCRE2_UCHAR buffer;
      //pcre2_get_error_message(pcre2_error_number_, &buffer, sizeof(buffer));
      //throw std::invalid_argument("Invalid special regex[" + special_pattern + "] Error[" + std::string((char*)buffer) + "]");
      throw std::invalid_argument("Invalid special regex[" + special_pattern + "]");
    }
    _special_regex_tls.resize(MAX_NUM_THREADS, nullptr);
    _special_match_data_tls.resize(MAX_NUM_THREADS);
    for (size_t i = 0; i < MAX_NUM_THREADS; ++i) {
      _special_regex_tls[i] = pcre2_code_copy(special_regex);
      pcre2_jit_compile(_special_regex_tls[i], PCRE2_JIT_COMPLETE);
      _special_match_data_tls[i] = pcre2_match_data_create_from_pattern(_special_regex_tls[i], NULL);
    }
    pcre2_code_free(special_regex);
  }
  
#if 0
  // Decoder
  LOG(INFO) << "CoreBPE::CoreBPE Init Decoder";
  for (const auto& kv : _encoder) {
    _decoder[kv.second] = kv.first;
  }
  for (const auto& kv : _special_tokens_encoder) {
    _special_tokens_decoder[kv.second] = std::vector<uint8_t>(kv.first.begin(), kv.first.end());
  }
        
  // Clone because I don't know how to tell Rust I'm not going to change the map
  LOG(INFO) << "CoreBPE::CoreBPE Init Sort Token";
  _sorted_token_bytes.reserve(_encoder.size());
  for (const auto& kv : _encoder) {
    _sorted_token_bytes.push_back(kv.first);
  }
  std::sort(_sorted_token_bytes.begin(), _sorted_token_bytes.end(), [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
  });
#endif
}

CoreBPE::~CoreBPE() {
  for (auto* regex : _regex_tls) {
    if (regex) {
      pcre2_code_free(regex);
    }
  }
  for (auto* regex : _special_regex_tls) {
    if (regex) {
      pcre2_code_free(regex);
    }
  }
  for (auto* match_data : _match_data_tls) {
    if (match_data) {
      pcre2_match_data_free(match_data);
    }
  }
  for (auto* match_data : _special_match_data_tls) {
    if (match_data) {
      pcre2_match_data_free(match_data);
    }
  }
}

//std::vector<size_t> CoreBPE::encode_ordinary(const std::string& text) const {
//}

std::vector<size_t> CoreBPE::encode(
  const std::string& text,
  const std::unordered_set<std::string>& allowed_special
  ) {
  return _encode_native(text, allowed_special).first;
}

/*
std::pair<std::vector<size_t>, TokenSeqSet> encode_with_unstable(
  const std::string& text,
  const std::unordered_set<std::string>& allowed_special
  ) const {
}

std::vector<size_t> encode_single_token(const std::vector<uint8_t>& piece) const {
}

std::vector<size_t> encode_single_piece(const std::vector<uint8_t>& piece) const {
}

std::vector<uint8_t> decode_bytes(const std::vector<size_t>& tokens) const {
}

std::vector<uint8_t> decode_single_token_bytes(size_t token) const {
}
*/
    
size_t CoreBPE::hash_current_thread() {
  auto tid = std::this_thread::get_id();
  std::hash<std::thread::id> hasher;
  return hasher(tid) % MAX_NUM_THREADS;
}
  
pcre2_code* CoreBPE::_get_tl_regex() const {
  return _regex_tls[hash_current_thread()];
}

pcre2_match_data* CoreBPE::_get_tl_match_data() const {
  return _match_data_tls[hash_current_thread()];
}

pcre2_code* CoreBPE::_get_tl_special_regex() const {
  return _special_regex_tls[hash_current_thread()];
}

pcre2_match_data* CoreBPE::_get_tl_special_match_data() const {
  return _special_match_data_tls[hash_current_thread()];
}

std::pair<std::vector<size_t>, size_t> CoreBPE::_encode_native(
  const std::string& text_ori,
  const std::unordered_set<std::string>& allowed_special
  ) {
  //LOG(INFO) << "All[" << text.size() << "]";
  //pcre2_code* special_regex = _get_tl_special_regex();
  //pcre2_code* normal_regex = _get_tl_regex();
  //pcre2_match_data* match_data = _get_tl_match_data();
  //pcre2_match_data* special_match_data = _get_tl_special_match_data();
  int slot_index = _slot.allocate();
  if (-1 == slot_index) {
    LOG(INFO) << "CoreBPE::_encode_native Allocate Index Failed";
    return {std::vector<size_t>(), -1};
  }
  pcre2_code* special_regex = _special_regex_tls[slot_index];
  pcre2_code* normal_regex = _regex_tls[slot_index];
  pcre2_match_data* match_data = _match_data_tls[slot_index];
  pcre2_match_data* special_match_data = _special_match_data_tls[slot_index];
  std::vector<size_t> ret;
  std::string_view text(text_ori);
  size_t last_piece_token_len = 0;
  size_t start = 0;
  while (true) {
    std::optional<std::pair<size_t, size_t>> next_special;
    // Find the next allowed special token, if any
    size_t start_find = start;
    while (true) {
      std::string_view search_text = text.substr(start_find);
      //LOG(INFO) << "1 => [" << text << "] Pos[" << (int)start_find << "] Search[" << search_text << "]";
      PCRE2_SPTR subject = (PCRE2_SPTR)search_text.data();
      PCRE2_SIZE subject_length = search_text.size();
      int rc = pcre2_match(
        special_regex,
        subject,
        subject_length,
        //start_find,
        0,
        PCRE2_NO_UTF_CHECK,
        special_match_data,
        NULL
        );
      if (rc > 0) {
        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(special_match_data);
        PCRE2_SIZE match_start = ovector[0];
        PCRE2_SIZE match_end = ovector[1];
        std::string_view matched = search_text.substr(match_start, match_end - match_start);
        //LOG(INFO) << "2 => " << match_start;
        if (allowed_special.find(std::string(matched.data(), matched.size())) != allowed_special.end()) {
          next_special = {start_find + match_start, start_find + match_end};
          break;
        }
        else {
          start_find = start_find + match_start + 1;
        }
      }
      else {
        //LOG(INFO) << "3 => ";
        break;
      }
    }
    size_t end = next_special ? next_special->first : text.size();
    //LOG(INFO) << "4 => " << (int)end;
    
    // Okay, here we go, compare this logic to _encode_ordinary_native
    if (start < end) {
      std::string_view sub = text.substr(start, end - start);
      //LOG(INFO) << "5 => " << sub;
      size_t start_offset = 0;
      PCRE2_SPTR subject = (PCRE2_SPTR)sub.data();
      PCRE2_SIZE subject_length = sub.size();
      while (true) {
        //LOG(INFO) << "AAAAAAA";
        int rc = pcre2_match(
          normal_regex,
          subject,
          subject_length,
          start_offset,
          PCRE2_NO_UTF_CHECK,
          match_data,
          NULL
          );
        if (rc > 0) {
          //LOG(INFO) << "6 => RC[" << rc << "]";
          PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);
          // [start0, end0, start1, end1, ...]
          PCRE2_SIZE match_start = ovector[0];
          PCRE2_SIZE match_end = ovector[1];
          if (match_end == match_start) {
            start_offset = match_start + 1;
            //LOG(INFO) << "Start == End";
          }
          else {
            start_offset = match_end;
            //LOG(INFO) << "Start != End";
          }
          //LOG(INFO) << "7 => " << match_start << "," << match_end << "," << start_offset;
          //std::string_view piece_str = sub.substr(match_start, match_end - match_start);
          std::string_view piece_str(sub.data() + match_start, match_end - match_start);
          //LOG(INFO) << "8 => " << piece_str;
          auto token_it = _encoder.find(piece_str);
          if (token_it != _encoder.end()) {
            //LOG(INFO) << "BBBB => ";
            ret.push_back(token_it->second);
            //LOG(INFO) << match_start << "," << token_it->second << "|" << ret.size(); 
            last_piece_token_len = 1;
            //LOG(INFO) << "BBBB => 1";
          }
          else {
            //LOG(INFO) << "CCC => ";
            auto tokens = byte_pair_encode(piece_str, _encoder);
            //LOG(INFO) << "CCC => 1";
            last_piece_token_len = tokens.size();
            //LOG(INFO) << "CCC => 2";
            ret.insert(ret.end(), tokens.begin(), tokens.end());
            //std::string tmp_str;
            //for (auto _ : tokens) {
            //  tmp_str += std::to_string(_);
            //  tmp_str += ",";
            //}
            //LOG(INFO) << match_start << "," << tmp_str << "|" << ret.size(); 
            //LOG(INFO) << "CCC => 3";
          }
        }
        else {
          break;
        }
      }
    }
    else {
      //throw std::runtime_error("Special Pos[" + std::to_string(start) + ">=" + std::to_string(end) + "]");
    }
    //LOG(INFO) << "9 => ";
    
    // And here we push the special token
    if (next_special) {
      //std::string_view special_str_view = text.substr(next_special->first, next_special->second - next_special->first);
      std::string_view special_str_view(text.data() + next_special->first, next_special->second - next_special->first);
      std::string special_str(special_str_view.data(), special_str_view.size());
      auto st_it = _special_tokens_encoder.find(special_str);
      if (st_it == _special_tokens_encoder.end()) {
        throw std::runtime_error("Special token not found: " + special_str);
      }
      ret.push_back(st_it->second);
      //LOG(INFO) << next_special->first << "," << st_it->second << "|" << ret.size();
      start = next_special->second;
      //LOG(INFO) << "Next[" << start << "]";
      last_piece_token_len = 0;
    }
    else {
      //LOG(INFO) << "Special Break";
      break;
    }
  }
  
  // last_piece_token_len is how many tokens came from the last regex split. This is used
  // for determining unstable tokens, since you can't merge across (stable) regex splits
  _slot.release(slot_index);
  return {ret, last_piece_token_len};
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
