数据准备  
data.sh/data.py

tokenizer测试  
perf_token.py

tiktoken测试  
perf_token_tiktoken.py

C++测试  
./test_token --gtest_filter=Token.QwenBatch

### Env
| Name | Content |
| :-----: | :----- |
| CPU | 20 * 12th Gen Intel(R) Core(TM) i7-12700 |
| File | /data/token_batch_data.jsonl |
| Line | 173964 |
| Token Count | 600872576 |

### Perf(Encode)
| Name              | Thread | Time(s) | Perf(TPS) | Multi |
| :-----:           | :----- | :-----  | :-----    | :-----  |
| Python(tokenizer) | 1      | 979     | 613762    | 0.09    |
| Python(tiktoken)  | 1      | 91      | 6602995   | 1       |
| C++               | 1      | 90      | 6676362   | 1.01    |
| C++               | 4      | 25      | 24034903  | 3.64    |
| C++               | 16     | 12      | 50072715  | 7.58    |
| C++               | 32     | 12      | 50072715  | 7.58    |

