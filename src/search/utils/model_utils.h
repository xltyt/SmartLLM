/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.09.16 15:47:54
 *
\****************************************************/

#ifndef _MODEL_UTILS_H__
#define _MODEL_UTILS_H__

#define C10_USE_GLOG
#include <torch/torch.h>
#include <glog/logging.h>

void load_module_from_safetensors(torch::nn::Module& module, const std::string& file_path);

struct TensorPrintOptions {
  int precision = 4;
  std::int64_t threshold = 1000;
  int edgeitems = 3;
  int linewidth = 80;
  bool sci_mode = false;
  };

std::string format_tensor(
  const torch::Tensor & tensor,
  const TensorPrintOptions & options = {}
  );

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
