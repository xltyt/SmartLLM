#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../../ref/')

from nanovllm.layers.layernorm import RMSNorm

import torch
import torch.nn.functional as F

hidden_size = 1024
batch_size = 2
seq_len = 10
eps = 1e-6
torch.manual_seed(42)

weight_generators = {
  #'ones': lambda: torch.ones(hidden_size),
  'normal': lambda: torch.randn(hidden_size),
  #'uniform': lambda: torch.rand(hidden_size) * 2 - 1,
  #'positive': lambda: torch.abs(torch.randn(hidden_size)) + 0.5,
  #'scaled': lambda: torch.randn(hidden_size) * 0.1 + 1.0,
}

with torch.no_grad():
  for weight_name, weight_func in weight_generators.items():
    # Input
    x = torch.rand(batch_size, seq_len, hidden_size)
    residual = torch.rand(batch_size, seq_len, hidden_size)
    
    # Init RMSNorm
    new_weight = weight_func()
    rms_norm = RMSNorm(hidden_size, eps)
    rms_norm.weight.data = new_weight

    # === Test 1
    output1 = rms_norm.forward(x.clone())
    
    # === Test 2
    output2, residual2 = rms_norm.forward(x.clone(), residual.clone())
    
    # === Test 3: Zero
    zero_tensor = torch.zeros(batch_size, seq_len, hidden_size)
    output_zero = rms_norm.forward(zero_tensor.clone())
    
    # === Test 4: Large
    large_tensor = torch.randn(batch_size, seq_len, hidden_size) * 1000.0
    output_large = rms_norm.forward(large_tensor.clone())
    
    # === Test 5: fp16
    x_fp16 = torch.rand(batch_size, seq_len, hidden_size, dtype=torch.float16)
    output_fp16 = rms_norm.forward(x_fp16.clone())

    torch.save(
      {
        "input": x,
        "weight": new_weight,
        "residual": residual,
        "output": output1,
        "output2": output2,
        "residual2": residual2,
        "input_zero": zero_tensor,
        "output_zero": output_zero,
        "input_large": large_tensor,
        "output_large": output_large,
        "input_fp16": x_fp16,
        "output_fp16": output_fp16,
      },
      "layer_norm.pt")
    print("Success[%s]" % weight_name)

# vim: set expandtab ts=4 sw=4 sts=4:
