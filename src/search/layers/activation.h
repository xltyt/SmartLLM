/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 
 *
\****************************************************/

#ifndef _ACTIVATION_H__
#define _ACTIVATION_H__

#define C10_USE_GLOG
#include <torch/torch.h>
#include <glog/logging.h>

class SiluAndMul : public torch::nn::Module {
public:
  torch::Tensor forward(torch::Tensor x);
};

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
