#include "crypt_utils.h"
#include <openssl/bn.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>
#include <string.h>
#include <fstream>

#define swap_byte(a, b) {swapByte = a; a = b; b = swapByte;}

void Crypt::_rc4Init(const void *binKey, uint16_t binKeySize, RC4KEY *key) {
  register unsigned char swapByte;
  register unsigned char index1 = 0, index2 = 0;
  unsigned char *state = &key->state[0];
  register uint16_t i;

  key->x = 0;
  key->y = 0;

  for (i = 0; i < 256; i++)
    state[i] = (unsigned char)i;
  for (i = 0; i < 256; i++) {
    index2 = (((unsigned char *)binKey)[index1] + state[i] + index2) & 0xFF;
    swap_byte(state[i], state[index2]);
    if(++index1 == binKeySize)index1 = 0;
  }
}

void Crypt::_rc4(void *buffer, uint32_t size, RC4KEY *key) { 
  register unsigned char swapByte;
  register unsigned char x = key->x;
  register unsigned char y = key->y;
  unsigned char *state = &key->state[0];

  for(register uint32_t i = 0; i < size; i++) {
    x = (x + 1) & 0xFF;
    y = (state[x] + y) & 0xFF;
    swap_byte(state[x], state[y]);
    ((unsigned char *)buffer)[i] ^= state[(state[x] + state[y]) & 0xFF];
  }

  key->x = x;
  key->y = y; 

}

void Crypt::_rc4Full(const void *binKey, uint16_t binKeySize, void *buffer, uint32_t size) {
  Crypt::RC4KEY key;
  Crypt::_rc4Init(binKey, binKeySize, &key);
  Crypt::_rc4(buffer, size, &key);
}
  
std::string Crypt::gen_random_string(int len) {
  std::string ret;
#if 0
  BIGNUM *rnd = BN_new();
  int length;
  char *show = NULL;
  int bits = len;
  int top = -1;
  int bottom = 0;
  BN_rand(rnd, bits, top, bottom);  
  length = BN_num_bits(rnd);
  show = BN_bn2hex(rnd);
  ret = show;
  OPENSSL_free(show);
  BN_free(rnd);
  return ret;
#else
  char *buf = new char[len + 1];
  buf[len] = 0;
  std::ifstream rfin("/dev/urandom");
  rfin.read(buf, len);
  rfin.close();
  for (int i = 0; i < len; i++) {
    char tmp_str[8];
    sprintf(tmp_str, "%02X", (unsigned char)buf[i]);
    ret += tmp_str;
  }
  delete buf;
  return ret;
#endif
}

std::string Crypt::base64_encode(const char* input, int length, bool with_new_line /*= false*/) {
  //创建一个base64对象
  BIO* b64 = BIO_new(BIO_f_base64());
  if (!with_new_line) {
    //取消'\n'。base64默认每64字节会添加一个‘\n’,包括末尾
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
  }
  //创建一个内存对象，用于存放编码之后的数据
  BIO* bmem = BIO_new(BIO_s_mem());
  //构成bio链，b64->bmem
  BIO_push(b64, bmem);
  //编码输入数据
  BIO_write(b64, input, length);
  //刷新bio链
  BIO_flush(b64);
  BUF_MEM* bptr = nullptr;
  //获取bio内存指针
  BIO_get_mem_ptr(b64, &bptr);
  //拷贝编码后的数据到用户空间，这里注意要加1，因为要向字符数组最后补充'\0'
  char *buff = new char[bptr->length + 1];
  memcpy(buff, bptr->data, bptr->length);
  //'\0'的ASCII码值是0
  buff[bptr->length] = 0;
  //释放资源
  BIO_free_all(b64);
  std::string resp(buff);
  delete[] buff;
  return resp;
}

std::string Crypt::base64_decode(const char* input, int length, bool with_new_line /*= false*/) {
  if (0 == length) {
    return "";
  }
  char *buffer = new char[length];
  //unique_ptr的构造函数不会对内存进行清零操作
  memset(buffer, 0, length);
  //创建一个base64对象
  BIO* b64 = BIO_new(BIO_f_base64());
  if (!with_new_line) {
    //取消'\n'。base64默认每64字节会添加一个‘\n’,包括末尾
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
  }
  //根据编码了的数据创建一个bio内存对象
  BIO* bmem = BIO_new_mem_buf(input, length);
  BIO_push(b64, bmem);
  BIO_flush(b64);
  //解码数据
  BIO_read(b64, buffer, length);
  //释放资源
  BIO_free_all(bmem);
  std::string resp(buffer);
  delete[] buffer;
  return resp;
}

/* vim: set expandtab nu smartindent ts=2 sw=2 sts=2: */
