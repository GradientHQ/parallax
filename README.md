<div align="center">
  <p align="center">
    <img src="https://raw.githubusercontent.com/GradientHQ/parallax/main/docs/images/parallax.png" width="720">
    <div align="center">
      <p style="font-size: 1.3em; font-weight: 600; margin-bottom: 10px;">Trusted by Partners</p>
      <img src="https://raw.githubusercontent.com/GradientHQ/parallax/main/docs/images/sglang.png" alt="SGLang" height="28" style="margin: 0 20px;">
      <img src="https://raw.githubusercontent.com/GradientHQ/parallax/main/docs/images/vllm.png" alt="vLLM" height="30" style="margin: 0 20px;">
      <img src="https://raw.githubusercontent.com/GradientHQ/parallax/main/docs/images/qwen.avif" alt="Qwen" height="30" style="margin: 0 20px;">
      <img src="https://raw.githubusercontent.com/GradientHQ/parallax/main/docs/images/deepseek.png" alt="DeepSeek" height="30" style="margin: 0 20px;">
      <img src="https://raw.githubusercontent.com/GradientHQ/parallax/main/docs/images/kimi.png" alt="Kimi" height="30" style="margin: 0 20px;">
      <img src="https://raw.githubusercontent.com/GradientHQ/parallax/main/docs/images/minimax.png" alt="Minimax" height="30" style="margin: 0 10px;">
      <img src="https://raw.githubusercontent.com/GradientHQ/parallax/main/docs/images/zai.svg" alt="ZAI" height="30" style="margin: 0 10px;"/>
    </div>
  </p>

[![license](https://img.shields.io/github/license/GradientHQ/parallax.svg)](https://github.com/GradientHQ/parallax/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/GradientHQ/parallax)](https://github.com/GradientHQ/parallax/issues)
[![open issues](https://img.shields.io/github/issues-raw/GradientHQ/parallax)](https://github.com/GradientHQ/parallax/issues)

<a href="https://www.producthunt.com/products/parallax-by-gradient?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-parallax&#0045;by&#0045;gradient" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1030922&theme=light&period=daily&t=1761986433128" alt="Parallax&#0032;by&#0032;Gradient - Host&#0032;LLMs&#0032;across&#0032;devices&#0032;sharing&#0032;GPU&#0032;to&#0032;make&#0032;your&#0032;AI&#0032;go&#0032;brrr | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>

</div>

| [**Gradient**](https://gradient.network)
| [**Blog**](https://gradient.network/blog/parallax-the-sovereign-ai-os)
| [**X(Twitter)(Gradient)**](https://x.com/Gradient_HQ)
| [**X(Twitter)(Parallax)**](https://x.com/tryParallax)
| [**Discord**](https://discord.gg/parallaxai)
| [**Arxiv**](https://arxiv.org/pdf/2509.26182v1)

## News
- [2026/6] 🚀 Parallax now supports ModelScope downloads!
- [2026/2] 🦞 Parallax now supports OpenClaw integration! See [Docs](./docs/user_guide/work_with_openclaw.md)
- [2025/10] 🔥 Parallax won #1 Product of The Day on Product Hunt!
- [2025/10] 🔥 Parallax version 0.0.1 has been released!

## About
A fully decentralized inference engine developed by [Gradient](https://gradient.network). Parallax lets you build your own AI cluster for model inference across distributed nodes despite their varying configuration and physical location. Its core features include:

- **Host local LLMs on personal devices**
- **Cross-platform support (Macs, Windows PCs, Linux/WSL GPU hosts)**
- **Pipeline-parallel model sharding**
- **Paged KV cache management and continuous batching for Mac**
- **Dynamic request scheduling and routing for high performance**
- **Local serving endpoint compatible with most agent frameworks**

The backend architecture:

* P2P communication powered by [Lattica](https://github.com/GradientHQ/lattica)
* GPU backend powered by [SGLang](https://github.com/sgl-project/sglang) and [vLLM](https://github.com/vllm-project/vllm)
* Mac backend powered by [MLX-LM](https://github.com/ml-explore/mlx-lm)

## Getting Started

### Step 0: Install Parallax

Choose the install path for the machine you are setting up.

| Environment | Recommended path |
|:------------|:-----------------|
| macOS Apple silicon | [Source install](./docs/user_guide/install.md#from-source) with `./install.sh` |
| Linux / WSL GPU host | [Source install](./docs/user_guide/install.md#from-source) with `./install.sh` |
| Windows | [Windows installer](./docs/user_guide/install.md#windows-application) |
| Linux GPU Docker | [Docker image](./docs/user_guide/install.md#docker) |

For macOS and Linux/WSL GPU hosts, source install is the main path:

```sh
git clone https://github.com/GradientHQ/parallax.git
cd parallax
./install.sh
source .venv/bin/activate
```

`install.sh` installs `uv` if needed, creates `.venv`, installs Parallax, and
builds the `vllm-rs` frontend binary into `.venv/bin`. It auto-detects macOS
and Linux extras; use `--extras` only when you want to override the default.

For Windows, Docker, Python version selection, and full prerequisites, see the
[Installation Guide](./docs/user_guide/install.md).

### Step 1: Launch the scheduler

Run this on the main machine that will host the scheduler and dashboard:

```sh
parallax run
```

### Step 2: Configure your model and cluster

Open [http://localhost:3001](http://localhost:3001), choose the model and
cluster size, then continue.

![Model config](https://raw.githubusercontent.com/GradientHQ/parallax/main/docs/images/node_config.png)

### Step 3: Join worker nodes

Copy the generated join command to each machine that will contribute compute.

```sh
parallax join
```

![Node join](https://raw.githubusercontent.com/GradientHQ/parallax/main/docs/images/node_join.png)

### Step 4: Chat or call the local API

Once the cluster is ready, use the chat interface or call the local
OpenAI-compatible endpoint at `http://localhost:3001/v1`.

![Chat interface](https://raw.githubusercontent.com/GradientHQ/parallax/main/docs/images/chat_interface.png)

For remote mode, no-frontend setup, standalone serving, ModelScope downloads,
and API examples, see the [Full Getting Started Guide](./docs/user_guide/quick_start.md).
To connect Parallax with OpenClaw, see [Working with OpenClaw](./docs/user_guide/work_with_openclaw.md).

## Supported Models

|              | Provider     | HuggingFace Collection  |  Blog  | Description |
|:-------------|:-------------|:----------------------------:|:----------------------------:|:----------------------------|
|DeepSeek      | Deepseek     | [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)<br>[DeepSeek-R1](https://huggingface.co/collections/deepseek-ai/deepseek-r1) <br>| [Deep Seek AI Launches Revolutionary Language Model](https://deepseek.ai/blog/deepseek-v32) | Deep Seek AI is proud to announce the launch of our latest language model, setting new standards in natural language processing and understanding. This breakthrough represents a significant step forward in AI technology, offering unprecedented capabilities in text generation, comprehension, and analysis. |
|MiniMax-M2    | MiniMax AI  | [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)<br>[MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1)<br>[MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) | [MiniMax M2.1: Significantly Enhanced Multi-Language Programming](https://www.minimax.io/news/minimax-m21) | MiniMax-M2.7 and MiniMax-M2.1 are enhanced sparse MoE models (about 230B parameters, 10B active) built for advanced coding and agentic workflows. They offer state-of-the-art intelligence, delivering efficient, reliable tool use and strong multi-step reasoning. |
|GLM           | Z AI | [GLM-5](https://huggingface.co/zai-org/GLM-5) <br>[GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) | [GLM-5.1 Overview](https://docs.z.ai/guides/llm/glm-5.1) | "GLM" is an advanced large language model series from Z AI, including GLM-5 and GLM-5.1. These models feature long-context support, strong coding and reasoning performance, enhanced tool-use and agent integration, and competitive results across leading open-source benchmarks. |
|Kimi-K2       | Moonshot AI  | [Kimi-K2](https://huggingface.co/collections/moonshotai/kimi-k2-6871243b990f2af5ba60617d) | [Kimi K2: Open Agentic Intelligence](https://moonshotai.github.io/Kimi-K2/) | "Kimi-K2" is Moonshot AI's Kimi-K2 model family, including Kimi-K2-Base, Kimi-K2-Instruct and Kimi-K2-Thinking. Kimi K2 Thinking is a state-of-the-art open-source agentic model designed for deep, step-by-step reasoning and dynamic tool use. It features native INT4 quantization and a 256k context window for fast, memory-efficient inference. Uniquely stable in long-horizon tasks, Kimi K2 enables reliable autonomous workflows with consistent performance across hundreds of tool calls.
|Qwen          | Qwen         | [Qwen3-Next](https://huggingface.co/collections/Qwen/qwen3-next-68c25fd6838e585db8eeea9d) <br>[Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) <br>[Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)| [Qwen3-Next: Towards Ultimate Training & Inference Efficiency](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) | The Qwen series is a family of large language models developed by Alibaba's Qwen team. It includes multiple generations such as Qwen2.5, Qwen3, and Qwen3-Next, which improve upon model architecture, efficiency, and capabilities. The models are available in various sizes and instruction-tuned versions, with support for cutting-edge features like long context and quantization. Suitable for a wide range of language tasks and open-source use cases. |
|gpt-oss       | OpenAI       | [gpt-oss](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) <br>[gpt-oss-safeguard](https://huggingface.co/collections/openai/gpt-oss-safeguard) | [Introducing gpt-oss-safeguard](https://openai.com/index/introducing-gpt-oss-safeguard/) | gpt-oss are OpenAI’s open-weight GPT models (20B & 120B). The gpt-oss-safeguard variants are reasoning-based safety classification models: developers provide their own policy at inference, and the model uses chain-of-thought to classify content and explain its reasoning. This allows flexible, policy-driven moderation in complex or evolving domains, with open weights under Apache 2.0. |
|Meta Llama 3  | Meta         | [Meta Llama 3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6) <br>[Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f) <br>[Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf) <br>[Llama 3.3](https://huggingface.co/collections/meta-llama/llama-33-67531d5c405ec5d08a852000) | [Introducing Meta Llama 3: The most capable openly available LLM to date](https://ai.meta.com/blog/meta-llama-3/) | "Meta Llama 3" is Meta's third-generation Llama model, available in sizes such as 8B and 70B parameters. Includes instruction-tuned and quantized (e.g., FP8) variants. |
|Step          | StepFun AI   | [Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) | [Step 3.5 Flash](https://static.stepfun.com/blog/step-3.5-flash/) | StepFun AI's Step-3.5-Flash is supported via a dedicated model architecture and is available as a pre-configured option in the scheduler UI. |

## Contributing

We warmly welcome contributions of all kinds! For guidelines on how to get involved, please refer to our [Contributing Guide](./docs/CONTRIBUTING.md).

## FAQ

<details>
<summary>When deploying on cloud servers, I encounter an error like <code>lattica RPC call failed</code>. What does this mean and how can I resolve it?</summary>

This error typically occurs when the necessary network ports for communication between the scheduler and nodes are blocked—most often due to firewall or security group settings on your cloud platform.

**How to fix:**
- Ensure that the relevant TCP/UDP ports for both the scheduler and nodes are open and accessible between all machines in your cluster.
    - By default, the scheduler uses HTTP port `3001`, and nodes use HTTP port `3000`. You can change these with the `--port` argument (e.g., `parallax run --port <your_port>` or `parallax join --port <your_port>`).
    - For Lattica (node-to-node) communication, random ports are used by default. It is best to explicitly specify which TCP and UDP ports to use (e.g., `--tcp-port <your_tcp_port> --udp-port <your_udp_port>`), and then open those ports for inbound and outbound traffic in your cloud provider's security settings.
- Check your cloud provider's firewall or network security group configurations:
    1. Open inbound rules for the ports mentioned above on all scheduler and node machines.
    2. Make sure that ports are open to the desired sources (e.g., to all cluster instances, or to your public IPs if required).

After updating the firewall/security group settings to allow these ports, restart your scheduler and nodes.

</details>

<details>
<summary>When running on macOS, I encounter the error <code>error sending packet on iface address No route to host (os error 65)</code>. What does this mean and how can I fix it?</summary>

On macOS, you need to allow your terminal or IDE (such as Terminal, iTerm2, VS Code, Cursor, etc.) access to the local network in order for Parallax to work correctly. If the application prompts you for network access the first time you run Parallax, click "Allow." If you have already denied access, follow these steps to enable it:

1. Open System Settings from the Apple menu.
2. Click on Privacy & Security in the sidebar.
3. Click on Local Network.
4. For each app listed, turn the ability to access your local network on or off using the toggle switch.

This will ensure Parallax has the proper network permissions for local communication.

</details>

<details>
<summary>When running the scheduler on Windows, nodes on other PCs cannot detect the scheduler ID over the local network. Why can't other machines join the cluster?</summary>

If you are running Parallax in WSL (Windows Subsystem for Linux), make sure you are using the "Mirrored" networking mode. By default, WSL uses "NAT" (Network Address Translation) mode, which isolates your WSL environment behind a virtual network. As a result, services running inside WSL (such as Parallax scheduler) are not directly accessible from other devices on the LAN.

To ensure that other machines on your network can connect to your WSL instance, change the WSL networking mode to "Mirrored" (supported on Windows 11 version 22H2 or later). In "Mirrored" mode, your WSL environment will share the same network as your host, allowing local network discovery and seamless joining of nodes to your Parallax cluster.

</details>
