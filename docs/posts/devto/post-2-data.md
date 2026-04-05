---
title: "Part 2: The Data — Building the First Public Coffee Roasting Audio Dataset"
published: false
description: "There was no public audio dataset for coffee roasting first crack detection anywhere on the internet. Before training anything, I had to build one from scratch — and the annotation and data engineering decisions made the difference between 87.5% precision and 100%."
tags: machinelearning, python, audio, huggingface
series: "From Prototype to Production in a Weekend"
cover_image:
---

There is no public audio dataset for coffee roasting first crack detection. Not on Hugging Face, not on Kaggle, not in any paper. Before the model in [Post 1](<!-- TODO: link -->) could train a single epoch, I had to build one from scratch — recording sessions, annotating audio in Label Studio, and designing a pipeline that prevents the leakage trap that silently inflates metrics in time-series audio ML.

The result is the [first public coffee roasting audio dataset](https://huggingface.co/datasets/syamaner/coffee-first-crack-audio): **973 annotated 10-second chunks from 15 roasting sessions**, open-sourced on Hugging Face. Two engineering decisions account for the model's **100% precision and 0 false positives**: recording-level data splitting, and class-weighted training against a 20/80 imbalance.

My role was the domain judgment: what to record, how to annotate, and which constraints to enforce. Oz built the pipeline — `convert_labelstudio_export.py`, `chunk_audio.py`, and `dataset_splitter.py` — from inside the Warp terminal as part of [PR #27](https://github.com/syamaner/coffee-first-crack-detection/pull/27). This post covers those decisions.

## The Gap

When an audio ML problem has no existing dataset, there are two paths: find adjacent data and hope it transfers, or build from scratch. Coffee roasting audio is domain-specific enough that generic kitchen sounds or industrial machine noise wouldn't transfer. A roaster produces a highly distinctive acoustic signature — sustained drum rotation, background heat noise, and then the sharp transient pops of first crack that vary by bean origin and roast profile.

I scoured the ecosystem hoping to save a weekend. A [HuggingFace Datasets search for "coffee"](https://huggingface.co/datasets?search=coffee) returns sentiment analysis, quality grading, and review corpora — zero audio. Kaggle and Papers with Code were identical dead ends. Because no one had ever published annotated first crack audio, there was no baseline to build on and no pre-existing validation to benchmark against.

This makes the [dataset](https://huggingface.co/datasets/syamaner/coffee-first-crack-audio) the primary open-source contribution of this project, more than the model weights. The model is useful to anyone who wants to replicate this setup. The dataset is useful to anyone who wants to build something different — a different architecture, a multi-event detector, or a roast profiling system — without having to spend weekends in front of a roaster first.

## Recording & Annotation

The nine legacy recordings were captured with a **FIFINE 669B USB condenser microphone** — a studio cardioid permanently mounted on the roaster. It is still the production sensor on the prototype today. The six newer recordings ([S17](https://github.com/syamaner/coffee-first-crack-detection/issues/24)) were captured with an **Audio-Technica ATR2100x** — a dynamic cardioid brought in specifically for data collection sessions. Both mics record via USB into Audacity: mono, 16-bit PCM, exported at 44.1kHz. The pipeline resamples to 16kHz for the `ASTFeatureExtractor`; full-resolution WAVs are gitignored and stored on Hugging Face.

The mic difference matters for the ML model. Condensers and dynamic mics produce measurably different acoustic signatures for the same event — different sensitivity, different frequency colouring, different noise floors. With 9 condenser recordings versus 6 dynamic in the training set, the model learned first crack primarily through the FIFINE's acoustic perspective. That imbalance directly explains the **27.4-second detection delay** on the mic-2 test recording versus near-instant detection on mic-1 — which is covered in [Post 3](<!-- TODO: link -->).

Each recording is a complete roast — 8 to 12 minutes — containing exactly one first crack event. That gives a single long audio file per session with a sustained background roast signature and one first crack window somewhere inside it.

### Annotation v1: The Fragmented Approach

For the first annotation pass in Label Studio, I annotated first crack the way it sounds: as individual events. I drew a region around each pop cluster — typically 1–6 seconds per region, tightly placed around each audible burst. The Brazil recordings from October 2025 show this clearly: 18 separate FC regions across a 90-second first crack window, ranging from 1.3 seconds (`chunk_011: 455.75–457.09s`) to 5.8 seconds, with gaps of up to 9 seconds between them.

![Label Studio v1 — individual pop-level annotations: narrow red first_crack bars scattered across the cracking period, with unlabeled gaps between each burst](../assets/screenshots/post-2-annotation-v1.png)

This produced baseline_v1's training data. The model trained, loss converged, and it hit 91.1% test accuracy. But it had **3 false positives** on the 45-sample test set — **87.5% precision**. At the time I attributed this to limited training data. That was wrong.

The actual problem was in how `chunk_audio.py` assigns labels. Every 10-second window gets a binary label based on whether its overlap with annotated `first_crack` regions meets a ≥50% threshold:

```python
# src/coffee_first_crack/data_prep/chunk_audio.py
def label_window(
    window_start: float,
    window_end: float,
    regions: list[dict[str, Any]],
    overlap_threshold: float = 0.5,
) -> str:
    window_duration = window_end - window_start
    overlap = compute_overlap(window_start, window_end, regions)
    if overlap >= overlap_threshold * window_duration:
        return "first_crack"
    return "no_first_crack"
```

With individual pop annotations, the very first cracking window in `brazil-3` is a 1.3-second region at 455.75s. A 10-second training window spanning 455–465s overlaps that annotation for only 1.3 seconds — **13% overlap** — and gets labeled `no_first_crack`. That window contains a real first crack pop. The same logic creates mislabeled windows at every inter-region gap throughout the recording.

### The Redesign: One Region Per Roast

The fix was conceptual, not algorithmic. First crack is not a collection of discrete events — it is a continuous acoustic period. It begins with the first audible pop and ends when cracking becomes infrequent enough to consider the phase over. The annotation should reflect the phase, not my subjective segmentation of burst clusters.

All 15 recordings were re-annotated with a single continuous region per roast, spanning from the first pop to the end of consistent cracking. The Label Studio config is deliberately minimal — one label, one region per file:

```xml
<View>
  <Audio name="audio" value="$audio" zoom="true" waveformHeight="100"/>
  <Labels name="label" toName="audio">
    <Label value="first_crack" background="#FF0000"/>
  </Labels>
</View>
```

Everything outside the annotated region is implicitly `no_first_crack`. One decision, enforced by the schema.

![Label Studio v2 — one continuous first_crack region covering the full cracking period. The "2" in the label panel is the label index, not a region count.](../assets/screenshots/post-2-label-studio.png)

With one continuous region, the ≥50% threshold in `label_window` becomes deterministic. Every 10-second window inside the first crack period gets clean overlap. There are no inter-region gaps to corrupt the training signal. Baseline_v2, trained on these re-annotated labels, produced **0 false positives** on a 191-sample test set — **100% precision**.

### The Pipeline Oz Built

With the annotation strategy locked, the problem shifted from domain judgment back to code. The Label Studio export is a single JSON file covering all recordings. Before any training can start, that file needs to be parsed per-recording, each recording chunked into 10-second windows, and the chunks split into train/val/test. I specified those constraints — fixed window size, ≥50% overlap threshold, recording-level split — and Oz implemented the three scripts that handle it, working from inside the Warp terminal as part of [PR #27](https://github.com/syamaner/coffee-first-crack-detection/pull/27):

- `convert_labelstudio_export.py` — parses the Label Studio JSON into per-recording annotation files
- `chunk_audio.py` — slides the fixed window, assigns labels via the overlap logic above
- `dataset_splitter.py` — splits at the recording level to prevent leakage (next section)

Here is Oz executing the full pipeline — Label Studio conversion, chunking all 973 segments, and recording-level split — before immediately invoking the `/train-model` skill:

{% embed <!-- TODO: Warp Block Link — PR #27 pipeline execution session --> %}

