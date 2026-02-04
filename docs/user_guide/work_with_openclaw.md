## Work with OpenClaw 🦞

### What is OpenClaw 🦞?

[OpenClaw](https://openclaw.ai/) is an open-source personal AI assistant that runs on your own machine. Unlike cloud-based AI services, OpenClaw gives you full control over your data and infrastructure.

Key features include:

- **Multi-platform chat integration**: Interact via WhatsApp, Telegram, Discord, Slack, Signal, or iMessage
- **Persistent memory**: Remembers your preferences and context across sessions
- **Full system access**: Read/write files, run shell commands, and control your browser
- **Extensible skills**: Use community-built skills or create your own
- **Model flexibility**: Works with Anthropic, OpenAI, or local models

Github repo of OpenClaw: https://github.com/openclaw/openclaw

### Prerequisites

To integrate Parallax with OpenClaw, you need to meet the prerequisites for both projects:

- **Node.js**: >= 22 (required by OpenClaw)
- **Python**: >=3.11 (required by Parallax)

Before proceeding, we assume you have already deployed Parallax on your AI cluster. For deployment instructions, please refer to:

- [Installation (Parallax)](./install.md)
- [Quick Start (Parallax)](./quick_start.md)


### Start your Parallax Service

**Step 1: Start the Scheduler**

On your scheduler machine, run:

```bash
parallax run --host 0.0.0.0
```

**Step 2: Select Model**

Open your browser and navigate to `localhost:3001` on the scheduler machine. Select your model and click **Continue**.

**Step 3: Start Edge Nodes**

On your edge nodes, run:

```bash
parallax join --max-sequence-length 65536 --max-num-tokens-per-batch 65536 --enable-prefix-cache
```

**Step 4: Test the Model**

On the scheduler machine, open your browser and navigate to `localhost:3001`. Use the chat interface to test if the model is working properly.

### Onboard OpenClaw

**Step 1: Install OpenClaw**

Use the official install script to install OpenClaw, skipping the onboard wizard:

```bash
curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard
```

**Step 2: Create Configuration File**

Create the configuration file at `~/.openclaw/openclaw.json` with the following content:

```json
{
  "agents": {
    "defaults": {
      "model": {
        "primary": "parallax/your-model-name"
      }
    }
  },
  "models": {
    "providers": {
      "parallax": {
        "baseUrl": "http://localhost:3001/v1",
        "apiKey": "placeholder",
        "api": "openai-completions",
        "models": [
          {
            "id": "your-model-name",
            "name": "Parallax Model"
          }
        ]
      }
    }
  }
}
```

**Step 3: Run Onboard**

```bash
openclaw onboard --install-daemon
```

During the onboard process:

1. Read and accept the OpenClaw risk disclaimer
2. When prompted for **onboarding mode**, select `Quick Start`
3. When prompted for **config handling**, select `Use existing values`
4. When prompted for **Model/auth provider**, select `Skip for now`
5. When prompted for **Filter models by provider**, select `All providers`
6. When prompted for **Default model**, select `Keep current (parallax/your-model-name)`
7. When prompted for **Select channel**, configure the channel based on your needs, or select `Skip for now`
8. When prompted for **Select skills**, configure the skills based on your needs, or select `Skip for now`
9. When prompted for **Enable hooks**, configure the hooks based on your needs, or select `Skip for now`
10. Wait a moment for Gateway services being installed.
11. When prompted for **How do you want to hatch your bot**, configure the way you hatch your bot based on your needs.

### Try on Browser

Open your browser and navigate to http://127.0.0.1:18789/. Start sending messages to OpenClaw and enjoy!

### Q&A

**Q: OOM Error**

```
libc++abi: terminating due to uncaught exception of type std::runtime_error: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

**A:** Add the `--kv-cache-memory-fraction` parameter when starting Parallax on edge nodes:

```bash
parallax join --max-sequence-length 65536 --max-num-tokens-per-batch 65536 --enable-prefix-cache --kv-cache-memory-fraction 0.5
```

If OOM errors persist, try using a smaller value for `--kv-cache-memory-fraction`.
