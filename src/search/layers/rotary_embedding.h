/****************************************************\
 *
 * Copyright (C) 2020 All Rights Reserved
 * Author       : xltyt@qq.com
 * Last modified: 2026.07.21 16:22:42
 *
\****************************************************/

#ifndef _ROTARY_EMBEDDING_H__
#define _ROTARY_EMBEDDING_H__

#define C10_USE_GLOG
#include <torch/torch.h>
#include <glog/logging.h>

class RotaryEmbedding : public torch::nn::Module {
public:
  RotaryEmbedding(
    int head_size,
    int rotary_dim,
    int max_position_embeddings,
    float base
    );
  virtual ~RotaryEmbedding();
    
public:
  std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor positions, torch::Tensor query, torch::Tensor key);
};

#endif

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
