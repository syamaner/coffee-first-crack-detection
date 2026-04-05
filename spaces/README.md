---
title: Coffee First Crack Detection
emoji: ☕
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "6.11.0"
app_file: app.py
pinned: false
license: apache-2.0
models:
  - syamaner/coffee-first-crack-detection
---

# ☕ Coffee First Crack Detection

Upload a 10-second coffee roasting audio clip to detect **first crack** — the critical moment when beans begin to pop.

**Model**: [syamaner/coffee-first-crack-detection](https://huggingface.co/syamaner/coffee-first-crack-detection)
**Code**: [github.com/syamaner/coffee-first-crack-detection](https://github.com/syamaner/coffee-first-crack-detection)

> **Read about the original fully working end-to-end prototype — inference, two MCP servers and an N8N agent:**
> - **Part 1** — [Training a Neural Network to Detect Coffee First Crack from Audio](https://dev.to/syamaner/part-1-training-a-neural-network-to-detect-coffee-first-crack-from-audio-an-agentic-development-1jei)
> - **Part 2** — [Building MCP Servers to Control a Home Coffee Roaster](https://dev.to/syamaner/part-2-building-mcp-servers-to-control-a-home-coffee-roaster-an-agentic-development-journey-with-58ik)
> - **Part 3** — [From Neural Networks to Autonomous Coffee Roasting: Orchestrating MCP Servers](https://dev.to/syamaner/part-3-from-neural-networks-to-autonomous-coffee-roasting-orchestrating-mcp-servers-with-net-58pd)

## How it works

An Audio Spectrogram Transformer (AST) fine-tuned on 973 labelled 10-second chunks from 15 coffee roast recordings returns `first_crack` / `no_first_crack` probabilities for the uploaded clip.

For full-recording sliding-window detection (with timeline chart) see the [source repo](https://github.com/syamaner/coffee-first-crack-detection).
