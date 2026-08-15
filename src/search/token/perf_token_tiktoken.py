#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import json
import unicodedata
import tiktoken
from tiktoken.load import data_gym_to_mergeable_bpe_ranks

lines = open("/data/token_batch_data.jsonl").readlines()
infos = []
for line in lines:
  obj = json.loads(line)
  infos.append({
    "text": unicodedata.normalize('NFC', obj["text"]),
    "ids": obj["ids"]
  })
print("Line[%d]" % len(infos))
mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
  "/data/Qwen3-0.6B/merges.txt",
  "/data/Qwen3-0.6B/vocab.json",
  None,
  None,
  True
  )
n_vocab = len(mergeable_ranks)
special_tokens = {}
obj = json.loads(open("/data/Qwen3-0.6B/tokenizer.json").read())
for item in obj["added_tokens"]:
  if not item["special"]:
    continue
  special_tokens[item["content"]] = item["id"]
  n_vocab += 1
encoder = tiktoken.Encoding(
    name="Qwen3",
    explicit_n_vocab=n_vocab,
    pat_str="(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens=special_tokens,
)
token_ct = 0
line_ct = 0;
time_start = time.time()
for info in infos:
  my_ids = encoder.encode(info["text"], allowed_special="all")
  if my_ids != info["ids"]:
    print("Invalid Ids[%d]" % line_ct)
    sys.exit(-1)
  token_ct += len(info["ids"])
  line_ct += 1
  #if line_ct >= 30000:
  #  break
time_used = time.time() - time_start
print("Time[%d]" % int(time_used))  
print("Count[%d]" % token_ct)  
