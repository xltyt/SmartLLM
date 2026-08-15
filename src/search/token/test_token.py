#!/usr/bin/python
# -*- coding: utf-8 -*-

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/data/Qwen3-0.6B/")

#prompts = [
#    "中国的首都是哪里",
#    "世界最高峰是哪个，比第二高峰高多少",
#]
#prompts = [
#    tokenizer.apply_chat_template(
#        [{"role": "user", "content": prompt}],
#        tokenize=False,
#        add_generation_prompt=True,
#    )
#    for prompt in prompts
#]
#for prompt in prompts:
#    print(prompt)
#    print(tokenizer.encode(prompt))

print(tokenizer.encode("Hello"))
print(tokenizer.encode("中国"))
print(tokenizer.encode("中国<|endoftext|>"))
print(tokenizer.encode("👍"))
print(tokenizer.encode("पता एडिट करना"))
print(tokenizer.encode("&amp;लेबल"))
print(tokenizer.encode("इस एड्रेस बुक से जुड़ा एड्रेस"))
print(tokenizer.encode("&amp;पता"))
print(tokenizer.encode("इस एड्रेस बुक से जुड़ी प्रविष्टि केवल भेजने वाले addresses के लिए बदली जा सकती है|"))
print(tokenizer.encode("नया स्वीकार्य पता"))
print(tokenizer.encode("नया भेजने वाला पता"))
print(tokenizer.encode("एडिट स्वीकार्य पता "))
print(tokenizer.encode("generating pronouns as the first word followed by \\texttt{<|endoftext|>}. Nonetheless, the trend of female pronouns being the least favorable is"))
print(tokenizer.encode("{<|endoftext|>}"))
print(tokenizer.encode("{<|endoftext|>}word followed by \\texttt{<|endoftext|>}. Nonetheless,"))
print("=========")

import json
import unicodedata
import tiktoken
from tiktoken.load import data_gym_to_mergeable_bpe_ranks
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
print(encoder.encode("Hello"))
# Unicode NFD
print(encoder.encode("इस एड्रेस बुक से जुड़ा एड्रेस"))
# Unicode NFC
print(encoder.encode(unicodedata.normalize('NFC', "इस एड्रेस बुक से जुड़ा एड्रेस")))
print(encoder.encode(unicodedata.normalize('NFC', "EuropeRussiaSt Petersburg\nHistoric Heart activities, tickets and more\n0 hours3 days\nSt. Petersburg Shore Excursion 2-Day City Group Tour with Visa\nThis city tour is the perfect way to explore St. Petersburg in 2 days. During this comprehensive group tour")))
print(encoder.encode(unicodedata.normalize('NFC', "generating pronouns as the first word followed by \\texttt{<|endoftext|>}. Nonetheless, the trend of female pronouns being the least favorable is"), allowed_special="all"))
print(encoder.encode(unicodedata.normalize('NFC', "{<|endoftext|>}"), allowed_special="all"))
print(encoder.encode(unicodedata.normalize('NFC', "{<|endoftext|>}word followed by \\texttt{<|endoftext|>}. Nonetheless,"), allowed_special="all"))

# vim: set expandtab ts=4 sw=4 sts=4:
