#include "sampler.h"

torch::Tensor Sampler::forward(torch::Tensor logits, torch::Tensor temperatures) {
	logits = logits.to(torch::kFloat32);
	
  auto temp_unsqueezed = temperatures.unsqueeze(1);
	logits = logits.div_(temp_unsqueezed);
  LOG(INFO) << logits;
	
  auto probs = torch::softmax(logits, -1);

	// Gumbel采样, 生成指数分布随机数，clamp到最小值
  LOG(INFO) << probs[0];
  LOG(INFO) << probs.sizes();
	auto noise = torch::empty_like(probs);
  LOG(INFO) << noise;
  torch::manual_seed(42);
	noise.exponential_(1.0);
  LOG(INFO) << noise;
	noise = noise.clamp_min_(1e-10);
  LOG(INFO) << noise;

	// probs / noise，然后取argmax
	auto sample_tokens = probs.div_(noise).argmax(-1);
  
  return sample_tokens;
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
