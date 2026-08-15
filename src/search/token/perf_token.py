#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/data/Qwen3-0.6B/")

lines = open("/data/token_batch_data.jsonl").readlines()
infos = []
for line in lines:
  obj = json.loads(line)
  infos.append({
    "text": obj["text"],
    "ids": obj["ids"]
  })
print("Line[%d]" % len(infos))
token_ct = 0
line_ct = 0;
time_start = time.time()
for info in infos:
  my_ids = tokenizer.encode(info["text"])
  if my_ids != info["ids"]:
    print("Invalid Ids")
    sys.exit(-1)
  token_ct += len(info["ids"])
  line_ct += 1
  #if line_ct >= 30000:
  #  break
time_used = time.time() - time_start
print("Time[%d]" % int(time_used))  
print("Count[%d]" % token_ct)  

