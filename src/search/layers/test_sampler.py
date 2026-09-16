#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../../ref/')

#from nanovllm.layers.sampler import Sampler

import torch
import torch.nn.functional as F
from torch import nn

class Sampler(nn.Module):
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        print(logits)
        probs = torch.softmax(logits, dim=-1)
        print(probs)
        print(probs.shape)

        noise = torch.empty_like(probs)
        print(noise)
        torch.manual_seed(42)
        noise = noise.exponential_(1)
        print(noise)
        noise = noise.clamp_min_(1e-10)
        print(noise)

        sample_tokens = probs.div_(noise).argmax(dim=-1)
        return sample_tokens

vocab_size = 1000
temps = torch.tensor([0.1, 10.0, 0.01, 100.0])

sampler = Sampler()
logits = torch.randn(temps.shape[0], vocab_size)
#torch.manual_seed(42)
output = sampler.forward(logits.clone(), temps.clone())

torch.save({"logits": logits, "temperatures": temps, "output": output}, "sampler.pt")
print("Success")
