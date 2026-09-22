#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

name = "/data/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
model.eval()

ids = tok("中国的首都是哪里，和美国的首都对比一下", return_tensors="pt").input_ids

captured = {}
def emb_hook(module, args, output):
  captured["hidden_states"] = args[0].detach()
  captured["position_ids"] = args[1].detach()
  captured["output_cos"] = output[0].detach()
  captured["output_sin"] = output[1].detach()
handle = model.model.rotary_emb.register_forward_hook(emb_hook)
with torch.inference_mode():
  _ = model(input_ids=ids, use_cache=False)

torch.save({
  "ids": ids,
  "hidden_states": captured["hidden_states"],
  "position_ids": captured["position_ids"],
  "output_cos": captured["output_cos"],
  "output_sin": captured["output_sin"]
  }, "test_rotary_emb.pt")

# vim: set expandtab ts=4 sw=4 sts=4:

