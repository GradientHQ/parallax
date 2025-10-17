<div align="center">
  <p align="center">
    <img src="docs/images/parallax.jpg" width="720">
  </p>
</div>

# Parallax
A fully decentralized inference engine developed by [Gradient](https://gradient.network). Parallax lets you build your own AI cluster for model inference onto a set of distributed nodes despite their varying configuration and physical location.
<!-- <h3> -->

| [**Gradient**](https://gradient.network)
| [**Blog**](https://gradient.network/blog/parallax-world-inference-engine)
| [**X(Twitter)**](https://x.com/Gradient_HQ)
| [**Discord**](https://discord.gg/gradientnetwork)
| [**Arxiv**](https://arxiv.org/pdf/2509.26182v1)

🔥 **NEW: Parallax version 0.0.1 has been released!**

<!-- </h3> -->

## Features
* Run LLM at home with personal devices.
* Cross-platform support.
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

Below are installation methods for different operating systems.

|  Operating System  |  Windows App  |  From Source | Docker |
|:-------------|:----------------------------:|:----------------------------:|:----------------------------:|
|Windows       | ✅️ | Not recommended | Not recommended |
|Linux | ❌️ | ✅️ | ✅️ |
|macOS | ❌️ | ✅️ | ❌️ |

### From Source
- For Linux/WSL (GPU):
```sh
git clone https://github.com/GradientHQ/parallax.git
cd parallax
pip install -e '.[gpu]'
```

- For macOS (Apple silicon):

We recommend macOS users to create an isolated Python virtual environment before installation.

```sh
git clone https://github.com/GradientHQ/parallax.git
cd parallax

# Enter Python virtual environment
python -m venv ./venv
source ./venv/bin/activate

pip install -e '.[mac]'
```

Next time to re-activate this virtual environment, run ```source ./venv/bin/activate```.

- Extra step for development:
```sh
pip install -e '.[dev]'
```

### Windows Application
[Click here](https://github.com/GradientHQ/parallax/releases/latest/download/Gradient_Parallax_PC_Setup.exe) to get latest Windows installer.

After installing .exe, right click Windows start button and click ```Windows Terminal(Admin)``` to start a Powershell console as administrator.

Start Windows dependencies installation by simply typing this command in console:
```sh
parallax install
```

Installation process may take around 30 minutes.

To see a description of all Parallax Windows configurations you can do:
```sh
parallax --help
```

### Docker
For Linux+GPU devices, Parallax provides a docker environment for quick setup. Choose the docker image according to the device's GPU architechture.

|  GPU Architecture  |  GPU Series  | Image Pull Command |
|:-------------|:----------------------------|:----------------------------|
|Blackwell       | RTX50 series/B100/B200... |```docker pull gradientservice/parallax:latest-blackwell```|
|Ampere/Hopper | RTX30 series/RTX40 series/A100/H100... |```docker pull gradientservice/parallax:latest-hopper```|

Run a docker container as below. Please note that generally the argument ```--gpus all``` is necessary for the docker to run on GPUs.
```sh
# For Blackwell
docker run -it --gpus all --network host gradientservice/parallax:latest-blackwell bash
# For Ampere/Hopper
docker run -it --gpus all --network host gradientservice/parallax:latest-hopper bash
```
The container starts under parallax workspace and you should be able to run parallax directly.

## Getting started

We will walk through you the easiest way to quickly set up your own AI cluster

### With Frontend

#### Step 1: Launch scheduler

First launch our scheduler on the main node, we recommend you to use your most convenient computer for this.
- For Linux/macOS:
```sh
parallax run
```

- For Windows, start Powershell console as administrator and run:
```sh
parallax run
```

#### Step 2: Set cluster and model config

Open http://localhost:3001 and you should see the setup interface.

![Model select](docs/images/node_join.png)

Select your desired node and model config and click continue.

#### Step 3: Connect your nodes

Copy the generated join command line to your node and run. For remote connection, you can find your scheduler-address in the scheduler logs.

```sh
# local area network env
parallax join
# public network env
parallax join -s {scheduler-address}
# example
parallax join -s 12D3KooWLX7MWuzi1Txa5LyZS4eTQ2tPaJijheH8faHggB9SxnBu
```

![Node join](docs/images/node_config.png)

You should see your nodes start to show up with their status. Wait until all nodes are successfully connected, and you will automatically be directed to the chat interface.

#### Step 4: Chat

Done! You have your own AI cluster now.

![Chat](docs/images/chat_interface.png)

### Without frontend
#### Step 1: Launch scheduler
First launch our scheduler on the main node.
```sh
parallax run -m {model-name} -n {number-of-worker-nodes}
```
For example:
```sh
parallax run -m Qwen/Qwen3-0.6B -n 2
```
Please notice and record the scheduler ip4 address generated in the terminal.

#### Step 2: Connect your nodes
For each distributed nodes including the main node, open a terminal and join the server with the scheduler address.
```sh
# local area network env
parallax join
# public network env
parallax join -s {scheduler-address}
```
For example:
```sh
# first node
parallax join -s 12D3KooWLX7MWuzi1Txa5LyZS4eTQ2tPaJijheH8faHggB9SxnBu
# second node
parallax join -s 12D3KooWLX7MWuzi1Txa5LyZS4eTQ2tPaJijheH8faHggB9SxnBu
```

#### Step 3: Call chat api with Scheduler
```sh
curl --location 'http://localhost:3001/v1/chat/completions' --header 'Content-Type: application/json' --data '{
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

### Skipping Scheduler
Developers can start Parallax backend engine without a scheduler. Pipeline parallel start/end layers should be set manually.
An example of serving Qwen3-0.6B with 2-nodes:
- First node:
```sh
python3 ./parallax/src/parallax/launch.py \
--model-path Qwen/Qwen3-0.6B \
--port 3000 \
--max-batch-size 8 \
--start-layer 0 \
--end-layer 14
```
- Second node:
```sh
python3 ./parallax/src/parallax/launch.py \
--model-path Qwen/Qwen3-0.6B \
--port 3000 \
--max-batch-size 8 \
--start-layer 14 \
--end-layer 28
```

Call chat API on one of the nodes:
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
### Uninstalling Parallax

For macOS or Linux, if you've installed Parallax via pip and want to uninstall it, you can use the following command:

```sh
pip uninstall parallax
```

For Docker installations, remove Parallax images and containers using standard Docker commands:

```sh
docker ps -a               # List running containers
docker stop <container_id> # Stop running containers
docker rm <container_id>   # Remove stopped containers
docker images              # List Docker images
docker rmi <image_id>      # Remove Parallax images
```

For Windows, simply go to Control Panel → Programs → Uninstall a program, find "Gradient" in the list, and uninstall it.

## Supported Models

|              | Provider     | HuggingFace Collection  |  Blog  | Description |
|:-------------|:-------------|:----------------------------:|:----------------------------:|:----------------------------|
|gpt-oss       | OpenAI       | [gpt-oss](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) | [Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/) | "gpt-oss" refers to OpenAI's open-source GPT models, including gpt-oss-20b and gpt-oss-120b. The number (e.g., 20b, 120b) indicates the parameter count (20 billion, 120 billion).  |
|Kimi-K2       | Moonshot AI  | [Kimi-K2](https://huggingface.co/collections/moonshotai/kimi-k2-6871243b990f2af5ba60617d) | [Kimi K2: Open Agentic Intelligence](https://moonshotai.github.io/Kimi-K2/) | "Kimi-K2" is Moonshot AI's Kimi-K2 model family, including Kimi-K2-Instruct and Kimi-K2-Instruct-0905. The models are designed for agentic intelligence and available in different versions and parameter sizes. |
|Qwen3-Next    | Qwen         | [Qwen3-Next](https://huggingface.co/collections/Qwen/qwen3-next-68c25fd6838e585db8eeea9d) | [Qwen3-Next: Towards Ultimate Training & Inference Efficiency](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) | "Qwen3-Next" is the latest generation of Qwen models by Alibaba/Qwen, with improved efficiency and performance. Includes models like Qwen3-Next-80B-A3B-Instruct (80B parameters, instruction-tuned) and Qwen3-Next-80B-A3B-Thinking (80B, reasoning enhanced). Variants include FP8 quantized and instruction-tuned models. |
|Qwen3         | Qwen         | [Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) | [Qwen3: Think Deeper, Act Faster](https://qwen.ai/blog?id=1e3fa5c2d4662af2855586055ad037ed9e555125&from=research.research-list) | "Qwen3" is the third generation of Qwen LLMs, available in multiple sizes (e.g., 0.6B, 1.7B, 4B, 8B, 14B, 30B, 32B, 235B). Variants include FP8 quantized and instruction-tuned models. |
|Qwen2.5       | Qwen         | [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) | [Qwen2.5: A Party of Foundation Models!](https://qwen.ai/blog?id=6da44b4d3b48c53f5719bab9cc18b732a7065647&from=research.research-list) | "Qwen2.5" is an earlier generation of Qwen models, with sizes like 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B. These models are available in base and instruction-tuned versions. |
|Meta Llama 3  | Meta         | [Meta Llama 3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6) <br>[Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f) <br>[Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf) <br>[Llama 3.3](https://huggingface.co/collections/meta-llama/llama-33-67531d5c405ec5d08a852000) | [Introducing Meta Llama 3: The most capable openly available LLM to date](https://ai.meta.com/blog/meta-llama-3/) | "Meta Llama 3" is Meta's third-generation Llama model, available in sizes such as 8B and 70B parameters. Includes instruction-tuned and quantized (e.g., FP8) variants. |
