// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include <string.h>
#include "hash_utils.h"
#include "murmurhash2.h"
#include <openssl/md5.h>

// The FALLTHROUGH_INTENDED macro can be used to annotate implicit fall-through
// between switch labels. The real definition should be provided externally.
// This one is a fallback version for unsupported compilers.
#ifndef FALLTHROUGH_INTENDED
#define FALLTHROUGH_INTENDED do { } while (0)
#endif

namespace common {

inline uint32_t DecodeFixed32(const char* ptr) {
#if (__BYTE_ORDER__) == (__ORDER_LITTLE_ENDIAN__)
    // Load the raw bytes
    uint32_t result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
#else
    return ((static_cast<uint32_t>(static_cast<unsigned char>(ptr[0])))
        | (static_cast<uint32_t>(static_cast<unsigned char>(ptr[1])) << 8)
        | (static_cast<uint32_t>(static_cast<unsigned char>(ptr[2])) << 16)
        | (static_cast<uint32_t>(static_cast<unsigned char>(ptr[3])) << 24));
#endif
}

uint32_t Hash(const char* data, size_t n, uint32_t seed /*= 0xbc9f1d34*/) {
  // Similar to murmur hash
  const uint32_t m = 0xc6a4a793;
  const uint32_t r = 24;
  const char* limit = data + n;
  uint32_t h = seed ^ (n * m);

  // Pick up four bytes at a time
  while (data + 4 <= limit) {
    uint32_t w = DecodeFixed32(data);
    data += 4;
    h += w;
    h *= m;
    h ^= (h >> 16);
  }

  // Pick up remaining bytes
  switch (limit - data) {
    case 3:
      h += static_cast<unsigned char>(data[2]) << 16;
      FALLTHROUGH_INTENDED;
    case 2:
      h += static_cast<unsigned char>(data[1]) << 8;
      FALLTHROUGH_INTENDED;
    case 1:
      h += static_cast<unsigned char>(data[0]);
      h *= m;
      h ^= (h >> r);
      break;
  }
  return h;
}

uint32_t Hash(const std::string& str, uint32_t seed /*= 0xbc9f1d34*/) {
  return Hash(str.c_str(), str.size(), seed);
}

uint64_t HashM(const std::string& str) {
  return MurmurHash2_64A(str.c_str(), str.size(), str.size());
}

std::string md5_string(const std::string& src) {
  //uint64_t sign[2];
  //MD5((const uint8_t *)src.c_str(), src.size(), (uint8_t *)sign);
  //char tmp[33] = {0};
  //snprintf(tmp, sizeof(tmp), "%016llX%016llX", sign[0], sign[1]);
  //return std::string(tmp);
  MD5_CTX ctx;
  unsigned char digest[16] = {0};
  MD5_Init(&ctx);
  MD5_Update(&ctx, src.c_str(), src.size());
  MD5_Final(digest, &ctx);
  
  char buf[33] = {0};  
  for (int i = 0; i < 16; i++) {
    char tmp[12] = {0};
    sprintf(tmp,"%02x", digest[i]);
    strcat(buf, tmp);
  }
  return std::string(buf);    
}

}  // namespace common
