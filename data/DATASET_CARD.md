---
dataset_info:
  features:
  - name: audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: label
    dtype: string
  - name: label_id
    dtype: int64
  - name: microphone
    dtype: string
  - name: coffee_origin
    dtype: string
  splits:
  - name: train
    num_examples: 587
  - name: val
    num_examples: 195
  - name: test
    num_examples: 191
task_categories:
  - audio-classification
language:
  - en
license: apache-2.0
tags:
  - coffee
  - roasting
  - first-crack
  - audio
  - spectrogram
pretty_name: Coffee First Crack Audio
size_categories:
  - n<1K
---

# Coffee First Crack Audio Dataset

Audio dataset for training coffee first crack detection models. Contains 10-second WAV chunks from coffee roasting recordings, labelled as **first_crack** (popping/cracking sounds) or **no_first_crack** (background roast noise).

> **Model**: [syamaner/coffee-first-crack-detection](https://huggingface.co/syamaner/coffee-first-crack-detection)
> **Source code**: [github.com/syamaner/coffee-first-crack-detection](https://github.com/syamaner/coffee-first-crack-detection)
>
> **Read about the original fully working end-to-end prototype — Inference, Two MCP Servers and N8N Agent:**
> - **Part 1** — [Training a Neural Network to Detect Coffee First Crack from Audio](https://dev.to/syamaner/part-1-training-a-neural-network-to-detect-coffee-first-crack-from-audio-an-agentic-development-1jei)
> - **Part 2** — [Building MCP Servers to Control a Home Coffee Roaster](https://dev.to/syamaner/part-2-building-mcp-servers-to-control-a-home-coffee-roaster-an-agentic-development-journey-with-58ik)
> - **Part 3** — [From Neural Networks to Autonomous Coffee Roasting: Orchestrating MCP Servers](https://dev.to/syamaner/part-3-from-neural-networks-to-autonomous-coffee-roasting-orchestrating-mcp-servers-with-net-58pd)

## Dataset Summary

- **973 total chunks** (fixed 10-second sliding windows, no overlap)
- **15 source recordings** from 2 microphones, 3 coffee origins
- **Recording-level split** (no data leakage between splits)
- **20% first_crack / 80% no_first_crack** — realistic class imbalance

| Split | first_crack | no_first_crack | Total | Recordings |
|-------|-------------|----------------|-------|------------|
| Train | 124 | 463 | 587 | 9 |
| Val | 37 | 158 | 195 | 3 |
| Test | 36 | 155 | 191 | 3 |

## Annotation Approach

Each source recording was annotated in Label Studio with a **single first_crack region** spanning from the first audible pop to the end of consistent cracking. The `chunk_audio.py` script then slid fixed 10-second windows across each recording and labelled each window based on overlap (>=50% threshold) with annotated first_crack regions.

This approach replaces the prototype method of manually annotating 20-30 small regions per file, producing consistent real-audio training chunks that match what the model sees during inference.

## Features

| Feature | Type | Description |
|---------|------|-------------|
| `audio` | Audio (16kHz) | 10-second mono WAV chunk |
| `label` | string | `first_crack` or `no_first_crack` |
| `label_id` | int | 1 = first_crack, 0 = no_first_crack |
| `microphone` | string | `mic-1-original` or `mic-2-new` |
| `coffee_origin` | string | e.g. `brazil`, `costarica-hermosa`, `brazil-santos` |

## Source Recordings

| Mic | Origin | Recordings | Notes |
|-----|--------|------------|-------|
| mic-1-original | costarica-hermosa | 5 | Legacy recordings from prototype |
| mic-1-original | brazil | 4 | Legacy recordings from prototype |
| mic-2-new | brazil | 4 | New recordings (Feb 2026) |
| mic-2-new | brazil-santos | 2 | New recordings (Apr 2026) |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("syamaner/coffee-first-crack-audio")
print(ds)
# DatasetDict({
#     train: Dataset({features: [audio, label, ...], num_rows: 587})
#     val: Dataset({features: [audio, label, ...], num_rows: 195})
#     test: Dataset({features: [audio, label, ...], num_rows: 191})
# })

# Access a sample
sample = ds["train"][0]
print(sample["label"], sample["microphone"], sample["coffee_origin"])
```

## Citation

```bibtex
@misc{yamaner2026coffeefc,
  author = {Yamaner, Sertan},
  title  = {Coffee First Crack Audio Dataset},
  year   = {2026},
  url    = {https://huggingface.co/datasets/syamaner/coffee-first-crack-audio}
}
```
