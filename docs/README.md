# 本地端到端验证指南（OpenAI 高级采样参数）

本文档用于在本地单机环境（单卡 4090 或 Mac/MLX）上验证 Parallax 的 OpenAI 高级采样参数（logprobs、logit_bias、penalties）及容错/降级机制。

---

## 1. 本地启动命令

### 1.1 入口与参数来源

- **入口**：`src/parallax/launch.py`（启动 Executor + HTTP Server + 可选 P2P）。
- **参数定义**：`src/parallax/server/server_args.py`（`--model-path`、`--port`、`--gpu-backend` 等）。
- **模型结构**：`src/parallax/models/` 下支持多种架构（如 `qwen2.py`、`qwen3.py`、`llama.py`、`deepseek_v2.py` 等）；本地单机时由 `launch.py` 拉取 config 并自动设置 `start_layer=0`、`end_layer=num_hidden_layers`。

### 1.2 推荐：最轻量模型 + 单机命令

**MLX（Mac / Apple Silicon）**  
使用小体积 MLX 模型，便于快速跑通并验证 **sampler 与 logprobs 全链路**（当前 logprobs 在 MLX executor 中实现）：

```bash
cd /Users/zicheng/Downloads/parallax-main

# 可选：使用虚拟环境
source .venv/bin/activate

# 单机启动（不指定 scheduler，自动单节点；HTTP 默认端口 3000）
python src/parallax/launch.py \
  --model-path mlx-community/Qwen2-0.5B-Instruct-4bit \
  --max-sequence-length 2048 \
  --max-batch-size 4 \
  --log-level INFO
```

若本地已有 HF 缓存或使用其他小模型（如 `Qwen/Qwen3-0.6B`），可将 `--model-path` 换成对应路径或 HuggingFace 名称；确保该模型在 `src/parallax/models/` 中有对应架构支持（如 Qwen2/Qwen3 等）。

**单卡 4090（CUDA）**  
同一套命令，在 CUDA 环境下会走 `--gpu-backend sglang`（默认）；需确保已安装 SGLang 等依赖。此时 logprobs/penalties 是否完整依赖 SGLang 侧实现；**高级参数与容错逻辑的完整验证建议在 MLX 路径下进行**。

```bash
# 同上，在 4090 机器上执行（自动检测 CUDA，使用 sglang）
python src/parallax/launch.py \
  --model-path mlx-community/Qwen2-0.5B-Instruct-4bit \
  --max-sequence-length 2048 \
  --max-batch-size 4 \
  --log-level INFO
```

### 1.3 端口与 URL

- HTTP 服务默认端口：**3000**（`server_args.py` 中 `--port` 默认值）。
- 若修改了 `--port`，下文所有请求 URL 需相应修改；测试脚本默认使用环境变量 `PARALLAX_BASE_URL`。

---

## 2. 基准测试验证（test_openai_advanced_params.py）

### 2.1 运行测试脚本

服务端已启动后，在**项目根目录**执行：

```bash
cd /Users/zicheng/Downloads/parallax-main

# 默认请求 http://127.0.0.1:8000，若服务在 3000 端口需显式指定
PARALLAX_BASE_URL=http://127.0.0.1:3000 python -m tests.test_openai_advanced_params
```

或先安装依赖再跑（脚本仅依赖 `requests`）：

```bash
pip install requests
PARALLAX_BASE_URL=http://127.0.0.1:3000 python -m tests.test_openai_advanced_params
```

### 2.2 脚本行为与 SSE 输出预期

- 脚本向 `{BASE_URL}/v1/chat/completions` 发送 **流式** 请求，body 包含：
  - `stream: true`
  - `logprobs: true`
  - `top_logprobs: 3`
  - `logit_bias: {"13": -100}`
  - `frequency_penalty: 1.2`
- 对 SSE 每行 `data: {...}` 解析后：
  - 若有 `choices[0].delta.content`，则**逐 token 打印 content**（不换行）；
  - 若有 `choices[0].logprobs.content` 且非空，则打印该 token 的 **logprob**、**token 字符串** 以及 **top_logprobs 条数**（例如 `[logprob=-0.5 token='Hello' top_logprobs=3 items]`）；
  - 若有 `choices[0].finish_reason`，则打印 `finish_reason`；
  - 最后打印 `[DONE]`、分隔线、**完整回复文本** 以及 “OK: stream with advanced params completed.”。

**预期终端输出格式示例**（片段）：

```
POST http://127.0.0.1:3000/v1/chat/completions
Request body: { ... }
------------------------------------------------------------
Hello
  [logprob=-0.12 token='Hello' top_logprobs=3 items]!
  [logprob=-0.34 token='!' top_logprobs=3 items]

finish_reason: stop
[DONE]
------------------------------------------------------------
Full reply: Hello!
OK: stream with advanced params completed.
```

重点验证：**流式 chunk 中应出现 `logprobs.content`**，且每个 content 项包含 `logprob`、`token` 以及约 3 个 `top_logprobs` 条目（与 `top_logprobs=3` 一致）。

---

## 3. 极客验证步骤（容错与降级）

### 3.1 无效 Token ID 拦截（logit_bias 越界）

**目的**：确认越界 `token_id` 被忽略并打 Warning，不抛异常、不崩进程。

**操作**：修改测试脚本中的 payload，将 `logit_bias` 改为一个**远超 vocab_size** 的 token id（例如 9999999）。例如在 `tests/test_openai_advanced_params.py` 里：

```python
# 原：logit_bias={"13": -100}
# 改为：
"logit_bias": {"9999999": 10.0},
```

保存后再次运行：

```bash
PARALLAX_BASE_URL=http://127.0.0.1:3000 python -m tests.test_openai_advanced_params
```

**服务端日志位置与预期**：

- 日志来自 **Executor 进程**（即跑 sampler 的进程）。若用 `launch.py` 单机启动，Executor 与 HTTP 同机，其 stdout/stderr 会与主进程一起输出到**启动 `launch.py` 的终端**（或你重定向到的日志文件）。
- 在 **sampler** 中，越界检查在 `src/parallax/server/sampling/sampler.py` 的 `_apply_logit_bias` 内；忽略越界项时会打出一条 **Warning**，内容类似：
  - `logit_bias: ignoring token_id=9999999 (out of vocab_size [0, 151936))`
- 预期：请求**正常完成**，返回 content 与（若开启）logprobs；**不应**出现未捕获异常或进程退出。

### 3.2 OOM / 异常降级（logprobs 计算失败仍返回 content）

**目的**：确认 logprobs 计算异常时，仅 logprobs 被跳过，主路 token 仍正常返回，且服务端打出预期 Warning。

**操作**：在 `src/parallax/server/sampling/sampler.py` 中，在 **logprobs 的 try 块内部**、在调用 `_log_softmax` 之后**立刻**插入一行 mock 异常（例如在约第 277 行，`log_probs = _log_softmax(...)` 之后）：

```python
        try:
            log_probs = _log_softmax(logits_scaled, axis=-1)
            raise RuntimeError("Mock OOM exception")   # 临时插入
            logprobs_info = _extract_logprobs_for_batch(
```

保存后重启服务，再次运行：

```bash
PARALLAX_BASE_URL=http://127.0.0.1:3000 python -m tests.test_openai_advanced_params
```

**预期**：

- **客户端**：流式响应**照常**收到，能逐 token 看到 `delta.content` 并拼出完整回复；但 **chunk 中不再带 `choices[0].logprobs`**（或 logprobs 为空），即“有 content，无 logprobs”。
- **服务端**：在 **Executor 进程** 的日志中应出现一条 **Warning**，由同一 try 的 `except` 打印，内容类似：
  - `logprobs computation failed (e.g. OOM), returning tokens without logprobs: Mock OOM exception`
  - 并伴随 `exc_info=True` 的 traceback。

验证完成后**务必删除**临时插入的 `raise RuntimeError("Mock OOM exception")`，恢复原逻辑。

---

## 4. 快速命令汇总（Copy-Paste）

```bash
# 1) 启动服务（项目根目录，MLX 轻量模型，端口 3000）
cd /Users/zicheng/Downloads/parallax-main
source .venv/bin/activate   # 可选
python src/parallax/launch.py \
  --model-path mlx-community/Qwen2-0.5B-Instruct-4bit \
  --max-sequence-length 2048 \
  --max-batch-size 4 \
  --log-level INFO

# 2) 另开终端：跑基准测试（端口 3000）
PARALLAX_BASE_URL=http://127.0.0.1:3000 python -m tests.test_openai_advanced_params

# 3) 无效 token_id 验证：改脚本里 logit_bias 为 {"9999999": 10.0} 后重跑上式，在启动服务的终端里找 "ignoring token_id"

# 4) OOM 降级验证：在 sampler.py 的 logprobs try 块内、_log_softmax 下一行加 raise RuntimeError("Mock OOM exception")，重启服务后重跑测试；客户端应有 content 无 logprobs，服务端有 "returning tokens without logprobs" 的 Warning
```

---

以上步骤覆盖：本地启动、基准流式测试与 logprobs/top_logprobs 输出预期、无效 token_id 拦截、以及 logprobs 异常降级行为，便于你在本地完成端到端与鲁棒性验证。
