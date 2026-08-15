/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.08.15 16:22:16
 *
\****************************************************/

#ifndef _CORE_BPE_H__
#define _CORE_BPE_H__

#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <stdint.h>
#include <atomic>
#include <absl/container/flat_hash_map.h>

std::string regex_escape(const std::string& text);

class SlotAllocator {
  static constexpr int NUM_SLOTS = 128;
  static constexpr int NUM_WORDS = NUM_SLOTS / 64;
  std::atomic<uint64_t> bitmap_[NUM_WORDS]{};

public:
  // 返回第一个空闲 slot，失败返回 -1
  int allocate();
  void release(int index);
};

class CoreBPE {
public:
  struct VectorHash {
    std::size_t operator()(const std::vector<uint8_t>& v) const noexcept;
    std::size_t operator()(size_t k) const noexcept;
  };

  struct VectorEqual {
    bool operator()(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) const noexcept;
    bool operator()(size_t a, size_t b) const noexcept;
  };

  struct string_hash {
    using is_transparent = void;
    std::size_t operator()(const char* str) const {
      return std::hash<std::string_view>{}(str);
    }
    std::size_t operator()(std::string_view str) const {
      return std::hash<std::string_view>{}(str);
    }
    std::size_t operator()(const std::string& str) const {
      return std::hash<std::string_view>{}(str);
    }
	};
	using string_equal = std::equal_to<>;

  template <typename K, typename V>
  //using HashMap = std::unordered_map<K, V, VectorHash, VectorEqual>;
  //using HashMap = std::map<K, V>;
  //using HashMap = absl::flat_hash_map<K, V, VectorHash, VectorEqual>;
  //using HashMap = std::unordered_map<K, V, string_hash, string_equal>;
  using HashMap = absl::flat_hash_map<K, V>;

  template <typename V>
  using StringMap = std::unordered_map<std::string, V>;
  
  struct VectorSizeTHash {
    std::size_t operator()(const std::vector<size_t>& v) const noexcept;
  };
  using TokenSeqSet = std::unordered_set<std::vector<size_t>, VectorSizeTHash>;

  static std::vector<uint32_t> decode_utf8(const std::string& text);
  // GPT-2
  static HashMap<std::vector<uint8_t>, size_t> data_gym_to_mergeable_bpe_ranks(
    const std::string& vocab_bpe_file,
    const std::string& encoder_json_file
    );
  // Qwen
  static HashMap<std::vector<uint8_t>, size_t> data_gym_to_mergeable_bpe_ranks(
    const std::string& tokenizer_file
    );
  // cl100k_base / whisper
  //static HashMap<std::vector<uint8_t>, size_t> load_tiktoken_bpe(const std::string& tiktoken_bpe_file);

public:
  CoreBPE(
    const HashMap<std::vector<uint8_t>, size_t>& encoder,
    const StringMap<size_t>& special_tokens_encoder,
    const std::string& pattern
    );
  virtual ~CoreBPE();
    
public:
  // Encode
  //std::vector<size_t> encode_ordinary(const std::string& text) const;
  std::vector<size_t> encode(
    const std::string& text,
    const std::unordered_set<std::string>& allowed_special
    );
  /*
  std::pair<std::vector<size_t>, TokenSeqSet> encode_with_unstable(
    const std::string& text,
    const std::unordered_set<std::string>& allowed_special
    ) const;
  std::vector<size_t> encode_single_token(const std::vector<uint8_t>& piece) const;
  std::vector<size_t> encode_single_piece(const std::vector<uint8_t>& piece) const;

  // Decode
  std::vector<uint8_t> decode_bytes(const std::vector<size_t>& tokens) const;
  std::vector<uint8_t> decode_single_token_bytes(size_t token) const;

  // Util
  std::vector<std::vector<uint8_t>> token_byte_values() const {
    return sorted_token_bytes_;
  }
  */
protected:
  static constexpr size_t MAX_NUM_THREADS = 128;
  static size_t hash_current_thread();
  pcre2_code* _get_tl_regex() const;
  pcre2_code* _get_tl_special_regex() const;
  pcre2_match_data* _get_tl_match_data() const;
  pcre2_match_data* _get_tl_special_match_data() const;

  std::pair<std::vector<size_t>, size_t> _encode_native(
    const std::string& text,
    const std::unordered_set<std::string>& allowed_special
    );

protected:
  // 内部数据结构
  HashMap<std::string, size_t> _encoder;
  StringMap<size_t> _special_tokens_encoder;
  // 普通 token -> bytes
  HashMap<size_t, std::vector<uint8_t>> _decoder;
  HashMap<size_t, std::vector<uint8_t>> _special_tokens_decoder;
  // 每线程正则副本 (普通)
  std::vector<pcre2_code*> _regex_tls;
  std::vector<pcre2_match_data*> _match_data_tls; // 每线程匹配数据
  // 每线程正则副本 (特殊)
  std::vector<pcre2_code*> _special_regex_tls;
  std::vector<pcre2_match_data*> _special_match_data_tls; // 每线程匹配数据
  // 所有 token 字节序排序
  std::vector<std::vector<uint8_t>> _sorted_token_bytes;
  // TLS占位
  SlotAllocator _slot;
};

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
