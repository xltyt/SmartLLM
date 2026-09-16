#include "rotary_embedding.h"

RotaryEmbedding::RotaryEmbedding(
  int head_size,
  int rotary_dim,
  int max_position_embeddings,
  float base
  ) {
}

RotaryEmbedding::~RotaryEmbedding() {
}
  
std::tuple<torch::Tensor, torch::Tensor> RotaryEmbedding::forward(torch::Tensor positions, torch::Tensor query, torch::Tensor key) {
  return std::make_tuple(positions, positions);
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
