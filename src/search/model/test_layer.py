#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

name = "/data/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, attn_implementation="eager")
model.eval()

LAYER_IDX = int(sys.argv[1])
target_layer = model.model.layers[LAYER_IDX]
#ids = tok("Hello", return_tensors="pt").input_ids
ids = tok("中国的首都是哪里，和美国的首都对比一下", return_tensors="pt").input_ids

attn_captured = {}
def attn_hook(module, args, kwargs, output):
  attn_captured["hidden_states"] = kwargs["hidden_states"].detach().clone()
  attn_captured["position_embeddings_cos"] = kwargs["position_embeddings"][0].detach().clone()
  attn_captured["position_embeddings_sin"] = kwargs["position_embeddings"][1].detach().clone()
  attn_captured["attention_mask"] = kwargs["attention_mask"].detach().clone()
  attn_captured["attn_output"] = output[0].detach()
  attn_captured["attn_weights"] = output[1].detach()
target_layer.self_attn.register_forward_hook(attn_hook, with_kwargs=True)

norm_captured = {}
def norm_hook(module, args, output):
  norm_captured["norm_input"] = args[0].detach()
  norm_captured["norm_output"] = output.detach()
target_layer.input_layernorm.register_forward_hook(norm_hook)

post_norm_captured = {}
def post_norm_hook(module, args, output):
  post_norm_captured["post_norm_input"] = args[0].detach()
  post_norm_captured["post_norm_output"] = output.detach()
target_layer.post_attention_layernorm.register_forward_hook(post_norm_hook)

mlp_captured = {}
def mlp_hook(module, args, output):
  mlp_captured["mlp_input"] = args[0].detach()
  mlp_captured["mlp_output"] = output.detach()
target_layer.mlp.register_forward_hook(mlp_hook)

layer_captured = {}
def layer_hook(module, args, output):
  layer_captured["layer_output"] = output.detach()
target_layer.register_forward_hook(layer_hook)

with torch.inference_mode():
  _ = model(input_ids=ids, use_cache=False)

torch.save({
  "ids": ids,
  "norm_input": norm_captured["norm_input"],
  "norm_output": norm_captured["norm_output"],
  "hidden_states": attn_captured["hidden_states"],
  "position_embeddings_cos": attn_captured["position_embeddings_cos"],
  "position_embeddings_sin": attn_captured["position_embeddings_sin"],
  "attention_mask": attn_captured["attention_mask"],
  "attn_output": attn_captured["attn_output"],
  "attn_weights": attn_captured["attn_weights"],
  "post_norm_input": post_norm_captured["post_norm_input"],
  "post_norm_output": post_norm_captured["post_norm_output"],
  "mlp_input": mlp_captured["mlp_input"],
  "mlp_output": mlp_captured["mlp_output"],
  "layer_output": layer_captured["layer_output"],
  },
  "test_layer_%d.pt" % LAYER_IDX)

