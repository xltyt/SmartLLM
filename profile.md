
# Profile (NPU)

启动服务时配置采集目录，例如：

~~~bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

./arm/bin/llm_service_arm_npu_9.0.0 \
  --model_path=/sharedata/liuyong/models/Qwen3-0.6B \
  --device=npu:0 --max_seq_len=2048 --decode_graph=true \
  --profile_dir=logs/profile
~~~

默认不启用 profiling，仅支持 NPU。开始采集、发送推理请求，然后停止：

~~~bash
curl -X POST http://127.0.0.1:28800/start_profile
curl -N http://127.0.0.1:28800/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"介绍一下人工智能","max_tokens":128,"stream":true,"ignore_eos":true}'
curl -X POST http://127.0.0.1:28800/stop_profile
~~~

开始、停止操作按 engine 队列顺序执行。停止会等待排在前面的推理、NPU 同步和采集收尾，完成后返回 HTTP 200。服务只生成原始采集数据，不调用 msprof。成功响应的 `data` 为本轮 session 的绝对路径，例如：

~~~json
{"status":0,"data":"/path/to/logs/profile/session-ABC123"}
~~~

停止后在服务器上手动导出：

~~~bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
msprof --export=on --output=/path/to/logs/profile/session-ABC123
~~~

`--output` 使用停止接口返回的目录，也可直接指定其中的 `PROF_*` 目录。结果生成在 `PROF_*/mindstudio_profiler_output/`，包含算子 CSV 和时间线 JSON。如需指定 Python，可在手动命令中增加 `--python-path=/path/to/python`。

每轮使用独立的 `session-*/PROF_*` 目录，可重复开始、停止。两个接口只接受 POST，否则返回 405；未配置采集目录、重复开始或未开始就停止返回 400。

`profile_dir` 的相对路径以服务工作目录为准。Profiling 会影响性能，基准测速时应关闭采集。
