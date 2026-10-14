# SmartLLM

# Compile
git submodule update --init --recursive  
make -C src 

# Perf(Qwen3-0.6B, 4070)

| Date | Content  | TTFT(ms) | TPOT(ms) | Prefill(tok/s) | Output(tok/s) |
|------|------|------|------|------|------|
| 2026-09-29 | 基础eager, kv cache, cpu, 无优化 | 102 | 49.16 | 196 | 20.34 |
| 2026-09-30 | 切换GPU | 357 | 5.86 | 56 | 170.6 |
| 2026-10-01 | GPU下加载改为fp16 | 506 | 5.88 | 64.5 | 170 |
| 2026-10-14 | arm机器 | 355 | 77.9 | 56.3 | 12.8 |
| 2026-10-14 | 支持910B | 312 | 25.22 | 64.1 | 39.6 |
