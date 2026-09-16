#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from safetensors.torch import load_file, save_file

class MyModel(torch.nn.Module):
    """Python equivalent of the C++ MyModel used by the alignment test."""

    def __init__(self, hidden_size: int = 4096, intermediate_size: int = 11008):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_size, intermediate_size)
        self.fc2 = torch.nn.Linear(intermediate_size, hidden_size)

    def forward_with_intermediates(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fc1_output = self.fc1(x)
        silu_output = torch.nn.functional.silu(fc1_output)
        output = self.fc2(silu_output)
        return fc1_output, silu_output, output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_intermediates(x)[-1]

def initialize_deterministically(model: MyModel, seed: int) -> None:
    """Initialize all parameters from one CPU generator with a stable order."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                torch.empty_like(parameter).uniform_(-0.05, 0.05, generator=generator)
            )

hidden_size = 4096
intermediate_size = 11008
batch_size = 3
seed = 20260915
root_dir = "../../../amd64/test/"

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
model = MyModel(hidden_size, intermediate_size).eval()
initialize_deterministically(model, seed)
weights = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
save_file(weights, root_dir + "model.safetensors")

input_generator = torch.Generator(device="cpu")
input_generator.manual_seed(seed + 1)
input_tensor = torch.randn(
    batch_size, hidden_size, generator=input_generator, dtype=torch.float32
)

with torch.inference_mode():
    fc1_output, silu_output, output = model.forward_with_intermediates(input_tensor)
torch.save({
  "hidden_size": hidden_size,
  "intermediate_size": intermediate_size,
  "batch_size": batch_size,
  "seed": seed,
  "input": input_tensor,
  "fc1_output": fc1_output,
  "silu_output": silu_output,
  "output": output
  }, root_dir + "model_utils.pt")

# vim: set expandtab nu ts=2 sw=2 sts=2:
