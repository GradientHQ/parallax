<div align="center">
  <p align="center">
    <img src="docs/images/parallax.png" width="720">
    <div align="center">
      <p style="font-size: 1.3em; font-weight: 600; margin-bottom: 10px;">Trusted by Partners</p>
      <img src="docs/images/qwen.avif" alt="Qwen" height="30" style="margin: 0 20px;">
      <img src="docs/images/sglang.png" alt="SGLang" height="28" style="margin: 0 20px;">
      <img src="docs/images/kimi.png" alt="Kimi" height="30" style="margin: 0 20px;">
      <img src="docs/images/minimax.png" alt="Minimax" height="30" style="margin: 0 10px;">
    </div>
  </p>

[![license](https://img.shields.io/github/license/GradientHQ/parallax.svg)](https://github.com/GradientHQ/parallax/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/GradientHQ/parallax)](https://github.com/GradientHQ/parallax/issues)
[![open issues](https://img.shields.io/github/issues-raw/GradientHQ/parallax)](https://github.com/GradientHQ/parallax/issues)

<a href="https://www.producthunt.com/products/parallax-by-gradient?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-parallax&#0045;by&#0045;gradient" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1030922&theme=light&period=daily&t=1761986433128" alt="Parallax&#0032;by&#0032;Gradient - Host&#0032;LLMs&#0032;across&#0032;devices&#0032;sharing&#0032;GPU&#0032;to&#0032;make&#0032;your&#0032;AI&#0032;go&#0032;brrr | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>

</div>

| [**Gradient**](https://gradient.network)
| [**Blog**](https://gradient.network/blog/parallax-the-sovereign-ai-os)
| [**X(Twitter)**](https://x.com/Gradient_HQ)
| [**Discord**](https://discord.gg/parallax)
| [**Arxiv**](https://arxiv.org/pdf/2509.26182v1)

## News
- [2025/10] 🔥 Parallax won #1 Product of The Day on Product Hunt!
- [2025/10] 🔥 Parallax version 0.0.1 has been released!

## About
A fully decentralized inference engine developed by [Gradient](https://gradient.network). Parallax lets you build your own AI cluster for model inference onto a set of distributed nodes despite their varying configuration and physical location. Its core features include:

- **Host local LLM on personal devices**
- **Cross-platform support**
- **Pipeline parallel model sharding**
- **Dynamic KV cache management & continuous batching for Mac**
- **Dynamic request scheduling and routing for high performance**

The backend architecture:

* P2P communication powered by [Lattica](https://github.com/GradientHQ/lattica)
* GPU backend powered by [SGLang](https://github.com/sgl-project/sglang)
* MAC backend powered by [MLX LM](https://github.com/ml-explore/mlx-lm)

## Installation

### Prerequisites
- Python>=3.11.0,<3.14.0
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
python3 -m venv ./venv
source ./venv/bin/activate

pip install -e '.[mac]'
```

Next time to re-activate this virtual environment, run ```source ./venv/bin/activate```.

- Extra step for development:
```sh
pip install -e '.[dev]'
```

**Note for macOS users regarding network permissions**

On macOS, you need to allow your terminal or IDE (such as Terminal, iTerm2, VS Code, Cursor, etc.) access to the local network in order for Parallax to work correctly. If the application prompts you for network access the first time you run Parallax, click "Allow." If you have already denied access, follow these steps to enable it:

1. Open System Settings from the Apple menu.
2. Click on Privacy & Security in the sidebar.
3. Click on Local Network.
4. For each app listed, turn the ability to access your local network on or off using the toggle switch.

This will ensure Parallax has the proper network permissions for local communication.


### Windows Application
[Click here](https://github.com/GradientHQ/parallax_win_cli/releases/latest/download/Parallax_Win_Setup.exe) to get latest Windows installer.

After installing .exe, right click Windows start button and click ```Windows Terminal(Admin)``` to start a Powershell console as administrator.

❗ Make sure you open your terminal with administrator privileges.
#### Ways to run Windows Terminal as administrator

- Start menu: Right‑click Start and choose “Windows Terminal (Admin)”, or search “Windows Terminal”, right‑click the result, and select “Run as administrator”.
- Run dialog: Press Win+R → type `wt` → press Ctrl+Shift+Enter.
- Task Manager: Press Ctrl+Shift+Esc → File → Run new task → enter `wt` → check “Create this task with administrator privileges”.
- File Explorer: Open the target folder → hold Ctrl+Shift → right‑click in the folder → select “Open in Terminal”.

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

## Getting Started

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

To allow the API to be accessible from other machines, add the argument `--host 0.0.0.0` when launching scheduler.
```sh
parallax run --host 0.0.0.0
```

When running `parallax run` for the first time or after an update, some basic info (like version and gpu name) might be sent to help improve the project. To disable this, use the `-u` flag:
```sh
parallax run -u
```

#### Step 2: Set cluster and model config

Open http://localhost:3001 and you should see the setup interface.

![Model select](docs/images/node_config.png)

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

![Node join](docs/images/node_join.png)

You should see your nodes start to show up with their status. Wait until all nodes are successfully connected, and you will automatically be directed to the chat interface.

When running `parallax join` for the first time or after an update, some basic info (like version and gpu name) might be sent to help improve the project. To disable this, use the `-u` flag:
```sh
parallax join -u
```

#### Step 4: Chat

Done! You have your own AI cluster now.

![Chat](docs/images/chat_interface.png)

#### Accessing the chat interface from another non-scheduler computer

You can access the chat interface from any non-scheduler computer, not just those running a node server. Simply start the chat server with:

```sh
# local area network env
parallax chat
# public network env
parallax chat -s {scheduler-address}
# example
parallax chat -s 12D3KooWLX7MWuzi1Txa5LyZS4eTQ2tPaJijheH8faHggB9SxnBu
```

After launching, visit [http://localhost:3002](http://localhost:3002) in your browser to use the chat interface.

To allow the API to be accessible from other machines, add the argument `--host 0.0.0.0` when launching chat interface.
```sh
parallax chat --host 0.0.0.0
```

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
|DeepSeek      | Deepseek     | [DeepSeek-V3.1](https://huggingface.co/collections/deepseek-ai/deepseek-v31) <br>[DeepSeek-R1](https://huggingface.co/collections/deepseek-ai/deepseek-r1) <br>[DeepSeek-V3](https://huggingface.co/collections/deepseek-ai/deepseek-v3) <br>[DeepSeek-V2](https://huggingface.co/collections/deepseek-ai/deepseek-v2) | [DeepSeek V3.1: The New Frontier in Artificial Intelligence](https://deepseek.ai/blog/deepseek-v31) | "DeepSeek" is an advanced large language model series from Deepseek AI, offering multiple generations such as DeepSeek-V3.1, DeepSeek-R1, DeepSeek-V2, and DeepSeek-V3. These models are designed for powerful natural language understanding and generation, with various sizes and capabilities for research and production use. |
|MiniMax-M2    | MiniMax AI  | [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) | [MiniMax M2 & Agent: Ingenious in Simplicity](https://www.minimax.io/news/minimax-m2) | MiniMax-M2 is a compact, fast, and cost-effective MoE model (230B parameters, 10B active) built for advanced coding and agentic workflows. It offers state-of-the-art intelligence and coding abilities, delivering efficient, reliable tool use and strong multi-step reasoning for developers and agents, with high throughput and low latency for easy deployment. |
|GLM-4.6       | Z AI | [GLM-4.6](https://huggingface.co/zai-org/GLM-4.6) | [GLM-4.6: Advanced Agentic, Reasoning and Coding Capabilities](https://z.ai/blog/glm-4.6) | GLM-4.6 improves upon GLM-4.5 with a longer 200K token context window, stronger coding and reasoning performance, enhanced tool-use and agent integration, and refined writing quality. Outperforms previous versions and is highly competitive with leading open-source models across coding, reasoning, and agent benchmarks. |
|Kimi-K2       | Moonshot AI  | [Kimi-K2](https://huggingface.co/collections/moonshotai/kimi-k2-6871243b990f2af5ba60617d) | [Kimi K2: Open Agentic Intelligence](https://moonshotai.github.io/Kimi-K2/) | "Kimi-K2" is Moonshot AI's Kimi-K2 model family, including Kimi-K2-Instruct and Kimi-K2-Instruct-0905. The models are designed for agentic intelligence and available in different versions and parameter sizes. |
|Qwen          | Qwen         | [Qwen3-Next](https://huggingface.co/collections/Qwen/qwen3-next-68c25fd6838e585db8eeea9d) <br>[Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) <br>[Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)| [Qwen3-Next: Towards Ultimate Training & Inference Efficiency](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) | The Qwen series is a family of large language models developed by Alibaba's Qwen team. It includes multiple generations such as Qwen2.5, Qwen3, and Qwen3-Next, which improve upon model architecture, efficiency, and capabilities. The models are available in various sizes and instruction-tuned versions, with support for cutting-edge features like long context and quantization. Suitable for a wide range of language tasks and open-source use cases. |
|gpt-oss       | OpenAI       | [gpt-oss](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) | [Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/) | "gpt-oss" refers to OpenAI's open-source GPT models, including gpt-oss-20b and gpt-oss-120b. The number (e.g., 20b, 120b) indicates the parameter count (20 billion, 120 billion).  |
|Meta Llama 3  | Meta         | [Meta Llama 3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6) <br>[Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f) <br>[Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf) <br>[Llama 3.3](https://huggingface.co/collections/meta-llama/llama-33-67531d5c405ec5d08a852000) | [Introducing Meta Llama 3: The most capable openly available LLM to date](https://ai.meta.com/blog/meta-llama-3/) | "Meta Llama 3" is Meta's third-generation Llama model, available in sizes such as 8B and 70B parameters. Includes instruction-tuned and quantized (e.g., FP8) variants. |
