/****************************************************\
 *
 * Copyright (C) 2019 All Rights Reserved
 * Last modified: 2026.09.16 15:48:41
 *
\****************************************************/

#include "model_utils.h"
#include <safetensors.hh>
#include <fstream>

static torch::ScalarType convert_dtype(safetensors::dtype dtype) {
  switch (dtype) {
    case safetensors::kFLOAT16:  return torch::kFloat16;
    case safetensors::kBFLOAT16: return torch::kBFloat16;
    case safetensors::kFLOAT32:  return torch::kFloat32;
    case safetensors::kFLOAT64:  return torch::kFloat64;
    case safetensors::kINT8:     return torch::kInt8;
    case safetensors::kINT16:    return torch::kInt16;
    case safetensors::kINT32:    return torch::kInt32;
    case safetensors::kINT64:    return torch::kInt64;
    case safetensors::kUINT8:    return torch::kUInt8;
    default:
    throw std::runtime_error("Unsupported dtype");
  }
}

void load_module_from_safetensors(torch::nn::Module& module, const std::string& file_path) {
  // Parse
  std::string warn;
  std::string err;
  safetensors::safetensors_t st;
  if (!safetensors::load_from_file(file_path, &st, &warn, &err)) {
    throw std::runtime_error("Failed to parse safetensors[" + err + "]");
  }
  if (!safetensors::validate_data_offsets(st, err)) {
    throw std::runtime_error("Invalid data_offsets[" + err + "]");
  }

  // Build Name => Tensor
  const uint8_t* data = st.mmaped ? st.databuffer_addr : st.storage.data();
  std::unordered_map<std::string, torch::Tensor> weight_map;
  const std::vector<std::string> tensor_keys = st.tensors.keys();
  for (int i = 0; i < st.tensors.size(); i++) {
    safetensors::tensor_t tensor_info;
    if (!st.tensors.at(i, &tensor_info)) {
      throw std::runtime_error("Failed to get safetensors[" + std::to_string(i) + "]");
    }
    const std::string& name = tensor_keys[i];
    
    // Shape
    std::vector<int64_t> shape(tensor_info.shape.begin(), tensor_info.shape.end());

    // Type
    auto dtype = convert_dtype(tensor_info.dtype);
    
    // Ptr
    const void* data_ptr = data + tensor_info.data_offsets[0];
    
    // Tensor
    auto tensor = torch::from_blob(
        const_cast<void*>(data_ptr),
        shape,
        torch::TensorOptions().dtype(dtype));
    
    weight_map[name] = tensor;
  }
  
  // Load
  torch::NoGradGuard no_grad;
  auto load_tensor = [&](const std::string& name, torch::Tensor dst) {
    auto it = weight_map.find(name);
    if (it == weight_map.end()) {
      throw std::runtime_error(name + " not found in safetensors");
    }
    if (it->second.sizes() != dst.sizes()) {
      throw std::runtime_error("shape mismatch: " + name);
    }
    dst.copy_(it->second.to(dst.dtype()));
  };

  for (auto& p : module.named_parameters()) {
    load_tensor(p.key(), p.value());
  }
  for (auto& b : module.named_buffers()) {
    if (weight_map.count(b.key())) {
      load_tensor(b.key(), b.value());
    }
  }
}

static std::string format_scalar(double value, const TensorPrintOptions & options, int width) {
  std::ostringstream stream;
  if (!std::isfinite(value)) {
    if (std::isnan(value)) {
      stream << "nan";
    } else {
      stream << (value > 0 ? "inf" : "-inf");
    }
  }
  else if (options.sci_mode) {
    stream << std::scientific << std::setprecision(options.precision) << value;
  }
  else {
    stream << std::fixed << std::setprecision(options.precision) << value;
  }
  const auto formatted = stream.str();
  if (static_cast<int>(formatted.size()) >= width) {
    return formatted;
  }
  return std::string(static_cast<std::size_t>(width - static_cast<int>(formatted.size())), ' ') + formatted;
}

static int scalar_width(double value, const TensorPrintOptions & options) {
  return static_cast<int>(format_scalar(value, options, 0).size());
}

static std::string format_tensor_recursive(
  const torch::Tensor & tensor,
  int indent,
  bool summarize,
  int width,
  const TensorPrintOptions & options
  ) {
  if (tensor.dim() == 0) {
    return format_scalar(tensor.item<double>(), options, width);
  }

  const auto size = tensor.size(0);
  const bool omit_middle = summarize && size > 2 * options.edgeitems;
  std::vector<std::string> parts;
  auto append_index = [&](std::int64_t index) {
    parts.push_back(
        format_tensor_recursive(tensor[index], indent + 1, summarize, width, options));
  };

  if (tensor.dim() == 1) {
    if (omit_middle) {
      for (int index = 0; index < options.edgeitems; ++index) {
        parts.push_back(format_scalar(tensor[index].item<double>(), options, width));
      }
      parts.emplace_back(" ...");
      for (int index = 0; index < options.edgeitems; ++index) {
        parts.push_back(
            format_scalar(tensor[size - options.edgeitems + index].item<double>(), options, width));
      }
    } else {
      for (std::int64_t index = 0; index < size; ++index) {
        parts.push_back(format_scalar(tensor[index].item<double>(), options, width));
      }
    }

    const int element_length = width + 2;
    const int elements_per_line = std::max(1, (options.linewidth - indent) / element_length);
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < parts.size(); ++index) {
      if (index > 0) {
        if (index % static_cast<std::size_t>(elements_per_line) == 0) {
          stream << ",\n" << std::string(static_cast<std::size_t>(indent + 1), ' ');
        } else {
          stream << ", ";
        }
      }
      stream << parts[index];
    }
    stream << ']';
    return stream.str();
  }

  if (omit_middle) {
    for (int index = 0; index < options.edgeitems; ++index) {
      append_index(index);
    }
    parts.emplace_back("...");
    for (int index = 0; index < options.edgeitems; ++index) {
      append_index(size - options.edgeitems + index);
    }
  } else {
    for (std::int64_t index = 0; index < size; ++index) {
      append_index(index);
    }
  }

  const std::string separator = "," + std::string(static_cast<std::size_t>(tensor.dim() - 1), '\n') +
    std::string(static_cast<std::size_t>(indent + 1), ' ');
  std::ostringstream stream;
  stream << '[';
  for (std::size_t index = 0; index < parts.size(); ++index) {
    if (index > 0) {
      stream << separator;
    }
    stream << parts[index];
  }
  stream << ']';
  return stream.str();
}

std::string format_tensor(
  const torch::Tensor & tensor,
  const TensorPrintOptions & options /*= {}*/
  ) {
  const auto cpu = tensor.detach().to(torch::kCPU).contiguous();
  if (cpu.numel() == 0) {
    return "tensor([])";
  }

  const auto values = cpu.to(torch::kFloat64).view({-1});
  int width = 1;
  for (std::int64_t index = 0; index < values.numel(); ++index) {
    width = std::max(width, scalar_width(values[index].item<double>(), options));
  }

  const bool summarize = cpu.numel() > options.threshold;
  std::ostringstream stream;
  stream << "tensor(" << format_tensor_recursive(cpu, 7, summarize, width, options) << ')';
  return stream.str();
}

/* vim: set expandtab nu ts=2 sw=2 sts=2: */
