/****************************************************\
 *
 * Copyright (C) 2012 All Rights Reserved
 * Last modified: 2025.05.07 17:45:10
 *
\****************************************************/

#ifndef _CRYPT_UTILS_H__
#define _CRYPT_UTILS_H__

#include <stdint.h>
#include <string>

namespace Crypt {
  typedef struct {      
    unsigned char state[256];       
    unsigned char x;        
    unsigned char y;
  }
  RC4KEY;
  void _rc4Init(const void *binKey, uint16_t binKeySize, RC4KEY *key);
  void _rc4(void *buffer, uint32_t size, RC4KEY *key);
  void _rc4Full(const void *binKey, uint16_t binKeySize, void *buffer, uint32_t size);
  std::string gen_random_string(int len);
  std::string base64_encode(const char* input, int length, bool with_new_line = false);
  std::string base64_decode(const char* input, int length, bool with_new_line = false);
}

#endif

/* vim: set expandtab nu smartindent ts=2 sw=2 sts=2: */
