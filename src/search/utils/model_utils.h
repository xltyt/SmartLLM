/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.10.15 11:54:08
 *
\****************************************************/

#ifndef _MODEL_UTILS_H__
#define _MODEL_UTILS_H__

#define C10_USE_GLOG
#include <torch/torch.h>
#include <glog/logging.h>

void load_module_from_safetensors(torch::nn::Module& module, const std::string& file_path, bool strict = true);

void load_module_from_safetensors_dir(torch::nn::Module& module, const std::string& dir_path, bool strict = true);

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

bool check_close(
  const std::string & name,
  const torch::Tensor & actual,
  const torch::Tensor & expected,
  double rtol,
  double atol
  );

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
