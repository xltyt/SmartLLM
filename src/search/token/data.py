#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/data/Qwen3-0.6B/")

f_write = open("/data/token_batch_data.jsonl", "w")
root_dir = "/data/token_batch_data/"

with open(root_dir + "custom.txt", encoding="utf-8") as f:
  for line in f:
    line=line.strip("\n")
    if not line or line.startswith("#"):
      continue
    f_write.write(json.dumps({
      "text": line,
      "ids": tokenizer.encode(line)
    }) + "\n")

with open(root_dir + "emoji-test.txt", encoding="utf-8") as f:
  for line in f:
    line=line.strip("\n")
    if not line or line.startswith("#"):
      continue
    code, rest = line.split(";",1)
    status = rest.split("#")[0].strip()
    if status != "fully-qualified":
        continue
    codepoints = code.strip().split()
    chars = ''.join(
      chr(int(cp,16))
      for cp in codepoints
      )
    #print(chars)
    f_write.write(json.dumps({
      "text": chars,
      "ids": tokenizer.encode(chars)
    }) + "\n")

for current_root, dirs, files in os.walk(root_dir):
  for filename in files:
    if filename.endswith('.jsonl'):
      file_path = os.path.join(current_root, filename)
      print(f"File: {file_path}")
      try:
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
          for line in f:
            line = line.strip("\n")
            if line:
              data = json.loads(line)
              f_write.write(json.dumps({
                "text": data["text"],
                "ids": tokenizer.encode(data["text"])
              }) + "\n")
            count += 1
            if count >= 10000:
              break
      except Exception as e:
          print(f"Read: {file_path}, Error: {e}")

# vim: set expandtab ts=2 sw=2 sts=2:
