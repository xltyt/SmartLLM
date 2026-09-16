/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 
 *
\****************************************************/

#include "activation.h"
  
torch::Tensor SiluAndMul::forward(torch::Tensor x) {
  auto chunks = x.chunk(2, -1);
  TORCH_CHECK(chunks.size() == 2, "Input must be splittable into 2 chunks");
  torch::Tensor x_part = chunks[0];
  torch::Tensor y      = chunks[1];
  return torch::silu(x_part) * y;
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
