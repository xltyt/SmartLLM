/****************************************************\
 *
 * Copyright (C) 2020 All Rights Reserved
 * Author       : xltyt@qq.com
 * Last modified: 2026.07.21 16:24:52
 *
\****************************************************/

#ifndef _SAMPLER_H__
#define _SAMPLER_H__

#define C10_USE_GLOG
#include <torch/torch.h>
#include <glog/logging.h>

class Sampler : public torch::nn::Module {
public:
  torch::Tensor forward(torch::Tensor logits, torch::Tensor temperatures);
};


#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
