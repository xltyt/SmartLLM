#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "/data/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
model.eval()

emb = model.model.embed_tokens
print(emb.weight.shape)                            # [151936, 1024]

ids = tok("Hello", return_tensors="pt").input_ids  # int64, [1, S]
with torch.no_grad():
  ref = emb(ids)                                   # [1, S, 1024]

torch.save({"ids": ids, "ref": ref, "weight": emb.weight.detach()}, "test_embed.pt")

# vim: set expandtab ts=4 sw=4 sts=4:
