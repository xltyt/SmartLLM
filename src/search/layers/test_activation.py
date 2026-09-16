#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../../ref/')

from nanovllm.layers.activation import SiluAndMul

import torch
import torch.nn.functional as F

torch.manual_seed(42)
x = torch.randn(2, 4, 8)

model = SiluAndMul()
model.eval()
with torch.no_grad():
  out = model(x)

torch.save({"input": x, "output": out}, "activation.pt")
print("Success")

# vim: set expandtab ts=4 sw=4 sts=4:
