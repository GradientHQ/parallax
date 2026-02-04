<div align="center">
  <p align="center">
    <img src="docs/images/parallax.png" width="720">
    <div align="center">
      <p style="font-size: 1.3em; font-weight: 600; margin-bottom: 10px;">Trusted by Partners</p>
      <img src="docs/images/sglang.png" alt="SGLang" height="28" style="margin: 0 20px;">
      <img src="docs/images/vllm.png" alt="vLLM" height="30" style="margin: 0 20px;">
      <img src="docs/images/qwen.avif" alt="Qwen" height="30" style="margin: 0 20px;">
      <img src="docs/images/deepseek.png" alt="DeepSeek" height="30" style="margin: 0 20px;">
      <img src="docs/images/kimi.png" alt="Kimi" height="30" style="margin: 0 20px;">
      <img src="docs/images/minimax.png" alt="Minimax" height="30" style="margin: 0 10px;">
      <img src="docs/images/zai.svg" alt="ZAI" height="30" style="margin: 0 10px;"/>
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
| [**Discord**](https://discord.gg/parallax)
| [**Arxiv**](https://arxiv.org/pdf/2509.26182v1)

## News
- [2026/2] 🦞 Parallax now supports OpenClaw integration! See [Docs](./docs/user_guide/work_with_openclaw.md)
- [2025/10] 🔥 Parallax won #1 Product of The Day on Product Hunt!
- [2025/10] 🔥 Parallax version 0.0.1 has been released!

## About
A fully decentralized inference engine developed by [Gradient](https://gradient.network). Parallax lets you build your own AI cluster for model inference onto a set of distributed nodes despite their varying configuration and physical location. Its core features include:

- **Host local LLM on personal devices**
- **Cross-platform support**
- **Pipeline parallel model sharding**
- **Paged KV cache management & continuous batching for Mac**
- **Dynamic request scheduling and routing for high performance**

The backend architecture:

* P2P communication powered by [Lattica](https://github.com/GradientHQ/lattica)
* GPU backend powered by [SGLang](https://github.com/sgl-project/sglang) and [vLLM](https://github.com/vllm-project/vllm)
* MAC backend powered by [MLX LM](https://github.com/ml-explore/mlx-lm)

## User Guide

- [Installation](./docs/user_guide/install.md)
- [Getting Started](./docs/user_guide/quick_start.md)
- [Working with OpenClaw 🦞](./docs/user_guide/work_with_openclaw.md)

## Contributing

We warmly welcome contributions of all kinds! For guidelines on how to get involved, please refer to our [Contributing Guide](./docs/CONTRIBUTING.md).

## Supported Models

|              | Provider     | HuggingFace Collection  |  Blog  | Description |
|:-------------|:-------------|:----------------------------:|:----------------------------:|:----------------------------|
|DeepSeek      | Deepseek     | [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)<br>[DeepSeek-R1](https://huggingface.co/collections/deepseek-ai/deepseek-r1) <br>| [Deep Seek AI Launches Revolutionary Language Model](https://deepseek.ai/blog/deepseek-v32) | Deep Seek AI is proud to announce the launch of our latest language model, setting new standards in natural language processing and understanding. This breakthrough represents a significant step forward in AI technology, offering unprecedented capabilities in text generation, comprehension, and analysis. |
|MiniMax-M2    | MiniMax AI  | [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)<br>[MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1) | [MiniMax M2.1: Significantly Enhanced Multi-Language Programming](https://www.minimax.io/news/minimax-m21) | MiniMax-M2.1 is an enhanced sparse MoE model (230B parameters, 10B active) built for advanced coding and agentic workflows. It offers state-of-the-art intelligence, delivering efficient, reliable tool use and strong multi-step reasoning. |
|GLM           | Z AI | [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) <br>[GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) | [GLM-4.7: Advancing the Coding Capability](https://z.ai/blog/glm-4.7) | "GLM" is an advanced large language model series from Z AI, including GLM-4.6 and GLM-4.7. These models feature long-context support, strong coding and reasoning performance, enhanced tool-use and agent integration, and competitive results across leading open-source benchmarks. |
|Kimi-K2       | Moonshot AI  | [Kimi-K2](https://huggingface.co/collections/moonshotai/kimi-k2-6871243b990f2af5ba60617d) | [Kimi K2: Open Agentic Intelligence](https://moonshotai.github.io/Kimi-K2/) | "Kimi-K2" is Moonshot AI's Kimi-K2 model family, including Kimi-K2-Base, Kimi-K2-Instruct and Kimi-K2-Thinking. Kimi K2 Thinking is a state-of-the-art open-source agentic model designed for deep, step-by-step reasoning and dynamic tool use. It features native INT4 quantization and a 256k context window for fast, memory-efficient inference. Uniquely stable in long-horizon tasks, Kimi K2 enables reliable autonomous workflows with consistent performance across hundreds of tool calls.
|Qwen          | Qwen         | [Qwen3-Next](https://huggingface.co/collections/Qwen/qwen3-next-68c25fd6838e585db8eeea9d) <br>[Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) <br>[Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)| [Qwen3-Next: Towards Ultimate Training & Inference Efficiency](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) | The Qwen series is a family of large language models developed by Alibaba's Qwen team. It includes multiple generations such as Qwen2.5, Qwen3, and Qwen3-Next, which improve upon model architecture, efficiency, and capabilities. The models are available in various sizes and instruction-tuned versions, with support for cutting-edge features like long context and quantization. Suitable for a wide range of language tasks and open-source use cases. |
|gpt-oss       | OpenAI       | [gpt-oss](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) <br>[gpt-oss-safeguard](https://huggingface.co/collections/openai/gpt-oss-safeguard) | [Introducing gpt-oss-safeguard](https://openai.com/index/introducing-gpt-oss-safeguard/) | gpt-oss are OpenAI’s open-weight GPT models (20B & 120B). The gpt-oss-safeguard variants are reasoning-based safety classification models: developers provide their own policy at inference, and the model uses chain-of-thought to classify content and explain its reasoning. This allows flexible, policy-driven moderation in complex or evolving domains, with open weights under Apache 2.0. |
|Meta Llama 3  | Meta         | [Meta Llama 3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6) <br>[Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f) <br>[Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf) <br>[Llama 3.3](https://huggingface.co/collections/meta-llama/llama-33-67531d5c405ec5d08a852000) | [Introducing Meta Llama 3: The most capable openly available LLM to date](https://ai.meta.com/blog/meta-llama-3/) | "Meta Llama 3" is Meta's third-generation Llama model, available in sizes such as 8B and 70B parameters. Includes instruction-tuned and quantized (e.g., FP8) variants. |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GradientHQ/parallax&type=timeline&logscale&legend=top-left)](https://www.star-history.com/#GradientHQ/parallax&type=timeline&logscale&legend=top-left)
