#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import time
import os

name = "/data/" + os.getenv("MODEL_NAME", "")
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, attn_implementation="eager")
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

reference_logits = []
def capture_logits(module, args, output):
  reference_logits.append(output.logits[0, -1, :].detach().cpu().clone())

hook = model.register_forward_hook(capture_logits)
time_start = time.time()
with torch.inference_mode():
  generate_ids = model.generate(ids, max_new_tokens=200, do_sample=False)
hook.remove()
time_used = time.time() - time_start
print(f"Time[{time_used}]")
text = tok.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(text)

assert len(reference_logits) == generate_ids.size(1) - ids.size(1)
torch.save({
  "ids": ids,
  "output": generate_ids,
  "max_length": 200
  },
  "test_runner.pt")
torch.save({
  "output": generate_ids,
  "logits": torch.stack(reference_logits),
  },
  "test_runner_logits.pt")
