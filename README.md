

# Parallax
A fully decentralized inference engine developed by [Gradient Network](https://gradient.network). Parallax reimagines model inference as a global, collaborative process—one where large language models are no longer chained to centralized infrastructure, but are instead decomposed, executed, and verified across a distributed machine mesh.

<h3>

[Gradient Network](https://gradient.network) | [Blog](https://gradient.network/blog/parallax-world-inference-engine) | [X(Twitter)](https://x.com/Gradient_HQ) | [Discord](https://discord.gg/gradientnetwork) | [arXiv]()

</h3>

## Features
* Run LLM at home with personal devices.
* Pipeline parallel model sharding.
* Dynamic KV cache management + continuous batching for MAC.
* Dynamic request scheduling and routing for high performance.

## Backend Architecture
* P2P communication powered by [Lattica](https://github.com/GradientHQ/lattica)
* GPU backend powered by [SGLang](https://github.com/sgl-project/sglang)
* MAC backend powered by [MLX LM](https://github.com/ml-explore/mlx-lm)

## Installation

### Prerequisites
- Python>=3.11.0
- Ubuntu-24.04 for Blackwell GPUs

### From Source
- For Linux/WSL (GPU):
```sh
git clone https://github.com/GradientHQ/parallax.git
cd parallax
pip install -e '.[gpu]'
```

- For macOS (Apple silicon):
```sh
git clone https://github.com/GradientHQ/parallax.git
cd parallax
pip install -e '.[mac]'
```

- Extra step for development:
```sh
pip install -e '.[dev]'
```

### Docker
For GPU devices, Parallax provides a docker environment for quick setup. Choose the docker image according to the device's GPU architechture.

|  GPU Architecture  |  GPU Series  | Image Pull Command |
|:-------------|:----------------------------|:----------------------------|
|Blackwell       | RTX50 series/B100/B200... |docker pull gradientservice/parallax:latest-blackwell|
|Ampere & Hopper | RTX30 series/RTX40 series/A100/H100... |docker pull gradientservice/parallax:latest-hopper|


## Usage on Distributed Devices
### Step 1: Launch scheduler
First launch our scheduler on the main node.
```sh
bash scripts/launch.sh -m {model-name} -n {number-of-worker-nodes}
```
For example:
```sh
bash scripts/launch.sh -m Qwen/Qwen3-0.6B -n 2
```
Please notice and record the scheduler ip4 address generated in the terminal.

### Step 2: Join each distributed nodes
For each distributed nodes including the main node, open a terminal and join the server with the scheduler address.
```sh
bash scripts/join.sh -m {model-name} -i {ip-address-of-current-node} -s {scheduler-address}
```
For example:
```sh
# first node
bash scripts/launch.sh -m Qwen/Qwen3-0.6B -i 192.168.1.1 -s /ip4/192.168.1.1/tcp/5001/p2p/xxxxxxxxxxxx
# second node
bash scripts/launch.sh -m Qwen/Qwen3-0.6B -i 192.168.1.2 -s /ip4/192.168.1.1/tcp/5001/p2p/xxxxxxxxxxxx
```

### Skipping Scheduler
Developers can start Parallax backend engine without a scheduler. Pipeline parallel start/end layers should be set manually.
An example of serving Qwen3-0.6B with 2-nodes:
- First node:
```sh
python3 ./parallax/src/parallax/launch.py \
--model-path Qwen/Qwen3-0.6B \
--port 3000 \
--dht-port 5000 \
--max-batch-size 8 \
--start-layer 0 \
--end-layer 14
```
- Second node:
```sh
python3 ./parallax/src/parallax/launch.py \
--model-path Qwen/Qwen3-0.6B \
--port 3000 \
--dht-port 5000 \
--max-batch-size 8 \
--start-layer 14 \
--end-layer 28 \
--initial-peers /ip4/192.168.1.1/tcp/5000/p2p/xxxxxxxxxxxx
```

## OpenAI Compatible API
Parallax starts OpenAI Compatible API either w/wo scheduler. Below is an example using CURL:
```sh
curl --location 'http://localhost:3000/v1/chat/completions' --header 'Content-Type: application/json' --data '{
    "max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": "hello"
      }
    ],
    "stream": true
}'
```

## Supported Models

|              |  HuggingFace  |  Blog  |
|:-------------|:----------------------------:|:----------------------------:|
|GPT-OSS       |[Link](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4)|[Link](https://openai.com/index/introducing-gpt-oss/)|
|Qwen3-Next    |[Link](https://huggingface.co/collections/Qwen/qwen3-next-68c25fd6838e585db8eeea9d)|[Link](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)|
|Qwen3         |[Link](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)|[Link](https://qwenlm.github.io/blog/qwen3/)|
|Qwen2.5       |[Link](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)|[Link](https://qwenlm.github.io/blog/qwen2.5/)|
|Llama3        |[Link](https://huggingface.co/meta-llama/collections)|[Link](https://ai.meta.com/blog/meta-llama-3/)|
