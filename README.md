# SmartLLM

# Compile
git submodule update --init --recursive  
make -C src 

# Perf(Qwen3-0.6B, 4070/910B2)

| Date | Content  | TTFT(ms) | TPOT(ms) | Prefill(tok/s) | Output(tok/s) |
|------|------|------|------|------|------|
| 2026-09-29 | 基础eager, kv cache, cpu, 无优化 | 102 | 49.16 | 196 | 20.34 |
| 2026-09-30 | 切换GPU | 357 | 5.86 | 56 | 170.6 |
| 2026-10-01 | GPU下加载改为fp16 | 506 | 5.88 | 64.5 | 170 |
| 2026-10-14 | arm机器 | 355 | 77.9 | 56.3 | 12.8 |
| 2026-10-14 | 支持910B | 312 | 25.22 | 64.1 | 39.6 |
| 2026-10-21 | 增加warmup，decode拿掉了logits回cpu做eval，修复之前ttft没有sync导致的计算不准的问题 | 31 | 28.78 | 645 | 34.7 |
| 2026-11-10 | 融合: qkv / 残差 + RMSNorm / gate&up + SwiGLU | 18 | 16.4 | 1111 | 60.9 |
| 2026-11-11 | 融合(以下性能多次求平均): 当前层mlp和下层input_layernorm | 17.5 | 15.9 | 1149.4 | 62.7 |
| 2026-11-11 | 融合: qkv融合的拷贝 | 18.4(性能波动) | 16.2 | 1087 | 61.9 |
| 2026-11-20 | kv形状固定 | 17.6 | 13.5 | 1136.4 | 74.1 |
| 2026-11-24 | decode支持graph | 18.6 | 8.4 | 1075.3 | 119.2 |
| **2026-11-26** | graph + KV 按行更新，替换为 `npu_scatter_nd_update_` | 18.6 | 4.02 | 1075.3 | 249.06 |

# Perf(HTTP, SmartLLM / vLLM-Ascend, 910B2)

| Date | Model | 实现 | 优化阶段 | CANN | 预热/测量 | TTFT(ms) | TPOT(ms) | Prefill(tok/s) | Output(tok/s) |
|------|------|------|------|------|------|------|------|------|------|
| 2026-11-26 | Qwen3-0.6B | SmartLLM | Graph 基线，KV 使用 `scatter_` | 9.0.0 | 2/5 | 23.19 | 8.62 | 22080.4 | 116.02 |
| 2026-11-26 | Qwen3-0.6B | SmartLLM | Graph + scatter 按行更新 | 9.0.0 | 2/5 | 19.99 | 4.18 | 25615.2 | 239.42 |
| 2026-11-26 | Qwen3-0.6B | vLLM-Ascend | Graph 参考基线 | 9.2.0 | 2/5 | 68.31 | 10.22 | 7495.1 | 97.83 |
| 2026-11-26 | Qwen3-4B | SmartLLM | Graph 基线，KV 使用 `scatter_` | 9.0.0 | 2/3 | 38.19 | 17.01 | 13407.0 | 58.77 |
| 2026-11-26 | Qwen3-4B | SmartLLM | Graph + scatter 按行更新 | 9.0.0 | 2/3 | 38.47 | 11.00 | 13308.4 | 90.91 |
| 2026-11-25 | Qwen3-4B | vLLM-Ascend | Graph 参考基线 | 9.2.0 | 3/10 | 79.38 | 10.88 | 6450.0 | 91.95 |

