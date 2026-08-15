// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// Simple hash function used for internal data structures

#ifndef STORAGE_LEVELDB_UTIL_HASH_H_
#define STORAGE_LEVELDB_UTIL_HASH_H_

#include <stddef.h>
#include <stdint.h>
#include <string>

namespace common {

uint32_t Hash(const char* data, size_t n, uint32_t seed = 0xbc9f1d34);
uint32_t Hash(const std::string& str, uint32_t seed = 0xbc9f1d34);
uint64_t HashM(const std::string& str);
std::string md5_string(const std::string& src);

}  // namespace common

#endif  // STORAGE_LEVELDB_UTIL_HASH_H_
