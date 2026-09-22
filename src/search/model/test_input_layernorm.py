#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

name = "/data/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
model.eval()

LAYER_IDX = int(sys.argv[1])
target_layer = model.model.layers[LAYER_IDX]
ids = tok("Hello", return_tensors="pt").input_ids

captured = {}

def norm_hook(module, args, output):
  captured["norm_input"] = args[0].detach()
  captured["norm_output"] = output.detach()
handle = target_layer.input_layernorm.register_forward_hook(norm_hook)
with torch.inference_mode():
  _ = model(input_ids=ids, use_cache=False)

#with torch.inference_mode():
#  hidden_states = model.model.embed_tokens(ids)
#  manual_norm_output = target_layer.input_layernorm(hidden_states)
#  captured["norm_input"] = hidden_states
#  captured["norm_output"] = manual_norm_output

torch.save({"ids": ids, "input": captured["norm_input"], "ref": captured["norm_output"]}, "test_input_layernorm_%d.pt" % LAYER_IDX)

# vim: set expandtab ts=4 sw=4 sts=4:
