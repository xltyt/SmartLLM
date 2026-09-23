#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

name = "/data/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, attn_implementation="eager")
model.eval()

#ids = tok("中国的首都是哪里，和美国的首都对比一下", return_tensors="pt").input_ids
#ids = tok("世界最高峰是哪里，简单介绍一下", return_tensors="pt").input_ids
messages = [{"role": "user", "content": "世界最高峰是哪里，简单介绍一下"}]
prompt_text = tok.apply_chat_template(
  messages,
  tokenize=False,
  add_generation_prompt=True,
  enable_thinking=False,
)
ids = tok(prompt_text, return_tensors="pt").input_ids

generate_ids = model.generate(ids, max_length=200, do_sample=False)
text = tok.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(text)

torch.save({
  "ids": ids,
  "output": generate_ids,
  "max_length": 100
  },
  "test_runner.pt")
