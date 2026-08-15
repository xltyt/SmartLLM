/****************************************************\
 *
 * Copyright (C) 2017 All Rights Reserved
 * Last modified: 2026.08.12 20:40:05
 *
****************************************************/

#ifndef _COMMON_H__
#define _COMMON_H__

#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <future>
#include <curl/curl.h>
#include <curl/easy.h>
#include <string>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <boost/filesystem.hpp>

#include <net_utils.h>
#include <string_utils.h>
#include <file_utils.h>
#include <timer.h>
#include <lock.h>

#define JSON_ADD_STRING(_obj, _name_key, _name, _val) \
  rapidjson::Value val_##_name_key(rapidjson::kStringType); \
  val_##_name_key.SetString(_val.c_str(), _val.size(), allocator_result); \
  _obj.AddMember(_name, val_##_name_key, allocator_result)

#define JSON_PUSH_STRING(_obj, _name_key, _val) \
  rapidjson::Value val_##_name_key(rapidjson::kStringType); \
  val_##_name_key.SetString(_val.c_str(), _val.size(), allocator_result); \
  _obj.PushBack(val_##_name_key, allocator_result)

#define JSON_ADD_INT(_obj, _name_key, _name, _val) \
  rapidjson::Value val_##_name_key(rapidjson::kNumberType); \
  val_##_name_key.SetInt(_val); \
  _obj.AddMember(_name, val_##_name_key, allocator_result)

#define JSON_ADD_UINT(_obj, _name_key, _name, _val) \
  rapidjson::Value val_##_name_key(rapidjson::kNumberType); \
  val_##_name_key.SetUint64(_val); \
  _obj.AddMember(_name, val_##_name_key, allocator_result)

#define JSON_ADD_DOUBLE(_obj, _name_key, _name, _val) \
  rapidjson::Value val_##_name_key(rapidjson::kNumberType); \
  val_##_name_key.SetDouble(_val); \
  _obj.AddMember(_name, val_##_name_key, allocator_result)

#define JSON_ADD_FLOAT(_obj, _name_key, _name, _val) \
  rapidjson::Value val_##_name_key(rapidjson::kNumberType); \
  val_##_name_key.SetFloat(_val); \
  _obj.AddMember(_name, val_##_name_key, allocator_result)

#define GET_JSONCPP_STRING(_root, _key, _default)  (_root.isMember(_key) && _root[_key].isString()) ? _root[_key].asString() : _default
#define GET_JSONCPP_INT(_root, _key, _default)   _root.isMember(_key) ?  ( _root[_key].isInt() ? _root[_key].asInt() : (_root[_key].isUInt() ? _root[_key].asUInt() : _default) ) : _default

#define GET_JSON_STRING(_root, _key, _default)  ((_root.HasMember(_key) && _root[_key].IsString()) ? _root[_key].GetString() : _default)
#define GET_JSON_STRING_BINARY(_root, _key, _default)  ((_root.HasMember(_key) && _root[_key].IsString()) ? std::string(_root[_key].GetString(), _root[_key].GetStringLength()) : _default)
#define GET_JSON_INT(_root, _key, _default)     ((_root.HasMember(_key) && _root[_key].IsInt()) ? _root[_key].GetInt() : _default)
#define GET_JSON_UINT(_root, _key, _default)    ((_root.HasMember(_key) && _root[_key].IsUint64()) ? _root[_key].GetUint64() : _default)
#define GET_JSON_FLOAT(_root, _key, _default)   ((_root.HasMember(_key) && _root[_key].IsFloat()) ? _root[_key].GetFloat() : _default)
#define GET_JSON_BOOL(_root, _key, _default)    ((_root.HasMember(_key) && _root[_key].IsBool()) ? _root[_key].GetBool() : _default)
#define GET_JSON_DOUBLE(_root, _key, _default)  ((_root.HasMember(_key) && _root[_key].IsDouble()) ? _root[_key].GetDouble() : _default)

#define REQUEST_JSON_PROCESS_START \
  rapidjson::Document doc_result; \
  rapidjson::Document::AllocatorType& allocator_result = doc_result.GetAllocator(); \
  rapidjson::Value json_val_result_items(rapidjson::kArrayType)
        
#define REQUEST_JSON_PROCESS_END \
  rapidjson::Value json_val_result(rapidjson::kObjectType); \
  JSON_ADD_INT(json_val_result, status, "status", 0); \
  json_val_result.AddMember("data", json_val_result_items, allocator_result); \
  rapidjson::StringBuffer sb; \
  rapidjson::Writer<rapidjson::StringBuffer> writer(sb); \
  json_val_result.Accept(writer); \
  resp_body = sb.GetString()

#define REQUEST_JSON_PROCESS_END_ITEMS \
  rapidjson::StringBuffer sb; \
  rapidjson::Writer<rapidjson::StringBuffer> writer(sb); \
  json_val_result_items.Accept(writer); \
  std::string resp_body = sb.GetString()

#if 1 == GTEST_PROTECED_USAGE
#define GTEST_PROTECED public
#else
#define GTEST_PROTECED protected
#endif

#ifndef SAFE_FREE
#define SAFE_FREE(ptr) if (ptr) { free(ptr); ptr = NULL; }
#endif
  
#ifndef SAFE_DELETE_AR
#define SAFE_DELETE_AR(ptr) if (ptr) { delete [] ptr; ptr = NULL; }
#endif
  
#ifndef SAFE_DELETE
#define SAFE_DELETE(ptr) if (ptr) { delete ptr; ptr = NULL; }
#endif

#ifndef _countof
#define _countof(_Array) (int)(sizeof(_Array) / sizeof(_Array[0]))
#endif

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
