#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

'''
Prefill 阶段明明需要因果遮蔽，传入的却是 None？
既然 Prefill 不能看未来词，为什么 Transformers 传给 self_attn 的也是 None？

核心原因：PyTorch 2.0+ 默认走的是 sdpa（Scaled Dot-Product Attention）
如果在加载模型时没有显式指定 attn_implementation="eager"，现代 Transformers 默认全部走 sdpa。
在 sdpa 模式下：
1. 如果输入没有 Padding（比如单条 Prompt 推理，所有 token 全有效）；
2. Transformers 为了节省显存和计算带宽，故意不生成那张庞大的下三角二维矩阵（避免在显存中开辟 $S \times S$ 的空间）；
3. 它把 attention_mask 设为 None，而在底层直接给 PyTorch C++ 内核传递了一个布尔标志：
   \(\text{F.scaled\_dot\_product\_attention}(Q, K, V, \text{attn\_mask}=\mathbf{None}, \mathbf{is\_causal=True})\)
4. 底层的 FlashAttention / Cutlass 算子看到 is_causal=True，在 GPU/NPU 计算时自动跳过上三角，数学上等价于加了下三角掩码，但在物理上根本没有这个张量。
'''

name = "/data/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, attn_implementation="eager")
model.eval()

LAYER_IDX = int(sys.argv[1])
target_layer = model.model.layers[LAYER_IDX]
#ids = tok("Hello", return_tensors="pt").input_ids
ids = tok("中国的首都是哪里，和美国的首都对比一下", return_tensors="pt").input_ids

captured = {}

def attn_hook(module, args, kwargs, output):
  captured["hidden_states"] = kwargs["hidden_states"].detach().clone()
  captured["position_embeddings_cos"] = kwargs["position_embeddings"][0].detach().clone()
  captured["position_embeddings_sin"] = kwargs["position_embeddings"][1].detach().clone()
  captured["attention_mask"] = kwargs["attention_mask"].detach().clone()
  captured["output_attn_output"] = output[0].detach()
  captured["output_attn_weights"] = output[1].detach()
handle = target_layer.self_attn.register_forward_hook(attn_hook, with_kwargs=True)
with torch.inference_mode():
  _ = model(input_ids=ids, use_cache=False)

torch.save({
  "ids": ids,
  "hidden_states": captured["hidden_states"],
  "position_embeddings_cos": captured["position_embeddings_cos"],
  "position_embeddings_sin": captured["position_embeddings_sin"],
  "attention_mask": captured["attention_mask"],
  "output_attn_output": captured["output_attn_output"],
  "output_attn_weights": captured["output_attn_weights"],
  }, "test_self_attn_0.pt")

# vim: set expandtab ts=4 sw=4 sts=4:
