#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

name = "/data/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, attn_implementation="eager")
model.eval()

ids = tok("中国的首都是哪里，和美国的首都对比一下", return_tensors="pt").input_ids

captured = {}
def model_hook(module, args, output):
  #print(output.keys())
  captured["last_hidden_state"] = output["last_hidden_state"].detach()
model.model.register_forward_hook(model_hook)

with torch.inference_mode():
  _ = model(input_ids=ids, use_cache=False)

torch.save({
  "ids": ids,
  "last_hidden_state": captured["last_hidden_state"],
  },
  "test_model.pt")

# vim: set expandtab ts=4 sw=4 sts=4:
