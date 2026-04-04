# Blog Strategy: Coffee First Crack Detection — Agentic HF Development Journey

## Context
* **What happened:** Built a complete HF-native audio ML pipeline — from scaffold to published model to RPi5 edge validation — across 15 stories in 5 phases, in ~2 days using Warp/Oz.
* **Existing POC series:** 3 dev.to posts covering an earlier prototype (no HF integration, no Pi deployment, 2 rough MCPs). 
    * [Part 1: Training](#)
    * [Part 2: MCP Servers](#)
    * [Part 3: .NET Aspire](#)
* **What's new:** HF-native model, published on Hub, ONNX INT8 edge deployment on RPi5, structured agentic dev workflow (`AGENTS.md`, epics, skills, rules, SSH), Copilot PR reviews.
* **Ambassador:** Warp's first ambassador cohort.

## Key Differentiation from POC Series
The POC series focused on **what** was built (model, MCPs, agent). This new series focuses on **how** it was built — the agentic development methodology itself. The story is: *"Here's how AI-assisted development with proper project structure lets you ship a production ML pipeline to edge hardware in a weekend."*

## Publishing Strategy
* **Dual-platform:** Each post published on both Dev.to and HuggingFace (community article at huggingface.co/new-blog).
    * **Dev.to:** Full narrative with agentic dev workflow emphasis, Warp ambassador angle.
    * **HF community blog:** Same core content, slightly more emphasis on HF ecosystem (Hub, Trainer API, model card, ONNX export), less Warp-specific tooling detail. HF articles are markdown with YAML frontmatter (`title`, `thumbnail`, `authors` with HF username).
* **Cross-linking:** Each post links to the other platform version, the HF model, the repo, and the POC series for background.

---

## Series: "From Training to Edge in a Weekend — Agentic ML Development"

### Post 1: Training & Publishing an HF-Native Audio ML Model with Agentic Development
**Core narrative:** How structured AI-agent collaboration (not just autocomplete) shipped a complete ML pipeline from zero to published HF model.

**Key sections:**
1.  **Intro:** Brief recap of the first crack detection problem (link to POC Part 1 for background). What's different now: HF-native, edge-ready, built with a structured agentic workflow.
2.  **The Agentic Development Setup:** * `AGENTS.md` as the single source of truth — architecture map, quick commands, platform notes, all in one file the agent reads before every task.
    * Epic state management (`docs/state/registry.md` → epic file → GitHub issue) — how the agent always knows what's done and what's next.
    * Skills (`.claude/skills/`) — repeatable workflows (train-model, evaluate-model, export-onnx, push-to-hub) the agent can invoke.
    * Rules — enforcing branch naming (`feature/{issue}-{slug}`), PR conventions, co-author attribution, coding standards (ruff, pyright).
    * One PR per story, branch protection on main.
3.  **The Build:** Walk through the 5 phases at a high level (not code-heavy — link to repo for details).
    * *Phase 1-2:* Scaffold → data pipeline → training (95.6% val accuracy, 208 training samples).
    * *Phase 3:* Inference with sliding window + pop-confirmation logic.
    * *Phase 4:* Publishing to HF Hub (model, model card, quickstart notebook).
4.  **Highlight:** GitHub Copilot as PR reviewer — how it caught real issues across PRs #17, #20, #23 (3 rounds of fixes on #23 alone).
5.  **Results:** 91.1% test accuracy / 0.913 F1 from 298 chunks (6 roasts), model live on HF Hub.
6.  **What made this fast:** The `AGENTS.md` + epics + skills pattern — agent never lost context, never went off-track, always knew the next step.
* **Dev.to angle:** Emphasize the Warp ambassador perspective, the developer experience, how this differs from "just using ChatGPT".
* **HF angle:** Emphasize the HF Trainer API, `save_pretrained`/`from_pretrained` pattern, model card best practices, quickstart notebook.

### Post 2: Deploying an Audio ML Model to Raspberry Pi 5 — The Edge Reality Check
**Core narrative:** Taking a cloud/desktop model to a $60 ARM board, validated entirely via SSH from Warp.

**Key sections:**
1.  **The SSH Workflow:** How Warp/Oz connected to RPi5 over SSH, ran benchmarks, debugged hardware issues — all from the same agentic conversation that built the model.
    * Agent created ONNX-only evaluation scripts (no PyTorch inference dependency on Pi).
    * Agent interpreted throttle flags (`0x50000`, `0xe0000`), diagnosed PSU and thermal issues.
    * Agent iterated on thread-limiting strategy based on live Pi results.
2.  **ONNX Export & Quantization:**
    * FP32 (345MB) → INT8 dynamic quantization (90MB) — zero quality loss.
    * Published both variants to HF Hub.
3.  **The Hardware Story (engaging content):**
    * Under-voltage crashes with standard USB-C chargers (even 96W Apple).
    * Thermal throttling without active cooling: 47°C → 77°C.
    * Thread-limiting as a survival strategy.
    * The $12 PSU fix.
4.  **Benchmarks (real numbers from the repo):**
    * Quality: 93.3% accuracy identical across Mac/Pi, FP32/INT8.
    * Latency breakdown: feature extraction 49ms (2%) vs ONNX inference 2,019ms (98%).
    * INT8 is free: 2x faster, 4x smaller, zero quality loss.
    * Production config: 2 threads, threshold=0.90, 2 cores free for MCP+UI.
    * The 10x gap: M-series Mac → ARM64 reality (197ms vs 2,452ms).
5.  **When to give up on edge:** AST at 87M params can't hit <500ms on Pi — options for the future (smaller models, knowledge distillation, offload to companion device).
6.  **Highlight:** Copilot review on edge code: How PR #23 went through 3 rounds of Copilot review fixes.
* **Dev.to angle:** The SSH story, hardware debugging narrative, agentic development on remote hardware.
* **HF angle:** ONNX export with `optimum`, publishing ONNX variants to Hub, `hf_hub_download()` for edge devices.

### Post 2.5 (NEW): Dataset Expansion, Hyperparameter Tuning & Detection Latency
**Core narrative:** How re-annotating 15 recordings with a simpler approach, building a proper chunking pipeline, and tuning hyperparameters produced a model that detects first crack in <1 second on known mics.

**Key sections:**
1. **The annotation problem:** Prototype used 20-30 small 3-5s regions per file → variable-length chunks → zero-padded to 10s during training → training/inference mismatch.
2. **The fix:** Single-region annotation (one `first_crack` region per roast) + sliding window chunker (`chunk_audio.py`) → consistent 10s real-audio chunks.
3. **Recording-level splitting:** Why chunk-level splitting causes data leakage (adjacent 10s windows from the same file are correlated) and how `dataset_splitter.py` groups by recording.
4. **Hyperparameter tuning story:**
   - Attempt 1 (lr=5e-5): oscillating loss, peaks at epoch 7 then overfits.
   - Diagnosis: lr too high for 587 training samples, loss spikes at epochs 3 and 5.
   - Attempt 2 (lr=2e-5, weight_decay 0.01→0.05, early_stop 5→3): stable training, early stops at epoch 5, best at epoch 2.
   - The lesson: 3x more data doesn't mean you can keep the same lr — smaller lr + stronger regularisation.
5. **Results comparison:**
   - v1: 91.1% test acc / 0.913 F1 / 3 FP (298 variable-length chunks, 6 roasts)
   - v2: **97.4% test acc / 0.925 F1 / 0 FP** (973 fixed 10s chunks, 15 roasts)
   - Precision: 87.5% → **100%** — zero false positives.
6. **Detection latency on full recordings:**
   - mic-1: 0.0s and 0.3s delay — near instant.
   - mic-2: 27.4s delay — model less confident on new microphone, first crack starts sparse.
   - The probability timeline visualisation (great for a blog chart).
7. **What the agent did:** Oz created the plan, built all 3 data_prep scripts, wrote 22 tests, updated docs, ran training, ran evaluation, ran full-recording inference — all in one conversation.
* **Dev.to angle:** The hyperparameter tuning workflow with an AI agent — how it diagnosed the oscillation, proposed the fix, and validated.
* **HF angle:** The `WeightedLossTrainer` handling 20/80 class imbalance, recording-level splitting best practice, sliding window chunking for audio.

### Post 3 (Optional/Future): "One MCP to Rule Them All" — Preview
* A shorter post or section at the end of Post 2 that previews the future roadmap:
* Single unified MCP server (not 2 like the POC) combining: FC detection, machine monitoring, machine control, statistics capture.
* Agentic UI and orchestration.
* Why one MCP was the lesson learned from the POC.
* *Note: Explicitly out of scope for this weekend — just a teaser.*

---

## Content Checklist per Post
- [ ] Include real code snippets from the repo (not synthetic examples).
- [ ] Link to HF model: [https://huggingface.co/syamaner/coffee-first-crack-detection](https://huggingface.co/syamaner/coffee-first-crack-detection)
- [ ] Link to GitHub repo: [https://github.com/syamaner/coffee-first-crack-detection](https://github.com/syamaner/coffee-first-crack-detection)
- [ ] Link to POC series for background context.
- [ ] Reference specific PRs and issues for credibility.
- [ ] Include actual metrics tables from `docs/state/epics/`.
- [ ] Mention Warp ambassador program (Dev.to versions).
- [ ] Cross-link between Dev.to and HF versions.
- [ ] YAML frontmatter for HF version (`title`, `thumbnail`, `authors: [{user: syamaner}]`).
- [ ] Compressed images/diagrams for both platforms.

---

## Screenshots to Capture
Before writing, scroll through Warp conversation history for the `coffee-first-crack-detection` project and screenshot these moments:

* **`AGENTS.md` read:** Agent reading the project rules at the start of a task.
* **TODO list + skill invocation:** Agent creating a task list and invoking a skill (`train-model`, `export-onnx`, etc.).
* **Epic state update:** Agent checking off a story in the epic doc after completing it.
* **SSH to RPi5:** The remote session showing benchmark output, throttle flag debugging, thermal readings.
* **Copilot PR review:** Review comments on a PR (e.g. #17 or #23) and the subsequent fix commits.
* **PR creation:** Agent pushing a branch and opening a PR with proper naming conventions from rules.
* **Test run:** `pytest` output showing all tests passing.
* *(Note: Ensure these are from the coffee project conversations specifically — not vinylid-ml or other repos).*

---

## Analysis Needed Before Writing
*Do this research when starting each post, not upfront:*

### For Post 1
* Review PR #16 (Phase 2&3) and PR #17 (Phase 3&4) Copilot review comments — what did Copilot catch? What categories of issues (type safety, edge cases, API misuse)?
* Review PR #18 (training validation) — the metrics story, how the model went from training to published.
* Review PR #20 (quickstart notebook) — Copilot review rounds.
* Count total Copilot review comments across all PRs and categorise them.
* Review `git log` to map the timeline: when did each phase start/finish? How does that support the "2 days" claim?

### For Post 2
* Review PR #23 in detail — 3 rounds of Copilot fixes, what changed each round.
* Extract the SSH session flow: what commands were run on the Pi, in what order.
* Map the hardware debugging timeline: when was the PSU issue discovered, when was thermal throttling identified, how did thread-limiting evolve.
* Pull exact benchmark numbers from `docs/state/epics/` (already documented there).
* Review `scripts/evaluate_onnx.py` and `scripts/benchmark_onnx_pi.py` for code snippets to include.

### For Both Posts
* Review the GitHub issue #1 epic structure — how stories were organised, what the checklist pattern looks like.
* Review `AGENTS.md` evolution — did it change across PRs? What was added as the project matured?
* Review `docs/state/` files to trace how state was maintained across sessions.

---

## Execution Order
1. **Posts 1 and 2 can be written in parallel** — they cover independent topics (training+publishing vs edge deployment). 
2. **Post 1 should be published first** since Post 2 references the model from Post 1.
3. Within each post, the Dev.to and HF versions can also be prepared in parallel (shared outline, platform-specific adjustments).



---------------------



# Blog Strategy: Coffee First Crack Detection — Agentic HF Development Journey

## Context
* **What happened:** Built a complete HF-native audio ML pipeline — from scaffold to published model to RPi5 edge validation — across 15 stories in 5 phases, in ~2 days using Warp/Oz.
* **Existing POC series:** 3 dev.to posts covering an earlier prototype (no HF integration, no Pi deployment, 2 rough MCPs). 
    * [Part 1: Training](#)
    * [Part 2: MCP Servers](#)
    * [Part 3: .NET Aspire](#)
* **What's new:** HF-native model, published on Hub, ONNX INT8 edge deployment on RPi5, structured agentic dev workflow (`AGENTS.md`, epics, skills, rules, SSH), Copilot PR reviews.
* **Ambassador:** Warp's first ambassador cohort.

## Key Differentiation from POC Series
The POC series focused on **what** was built (model, MCPs, agent). This new series focuses on **how** it was built — the agentic development methodology itself. The story is: *"Here's how AI-assisted development with proper project structure lets you ship a production ML pipeline to edge hardware in a weekend."*

## Publishing Strategy
* **Dual-platform:** Each post published on both Dev.to and HuggingFace (community article at huggingface.co/new-blog).
    * **Dev.to:** Full narrative with agentic dev workflow emphasis, Warp ambassador angle.
    * **HF community blog:** Same core content, slightly more emphasis on HF ecosystem (Hub, Trainer API, model card, ONNX export), less Warp-specific tooling detail. HF articles are markdown with YAML frontmatter (`title`, `thumbnail`, `authors` with HF username).
* **Cross-linking:** Each post links to the other platform version, the HF model, the repo, and the POC series for background.
* **Data Provenance:** Every metric referenced in the series (accuracy, latency, F1) must include a hyperlink directly to the specific Git commit hash or evaluation JSON file in the repo to prove authenticity.
* **Tooling Distinction:** Explicitly define the AI boundaries early on: **Warp/Oz** is the "Doer" (writing code, running SSH, executing skills), while **GitHub Copilot** is the "Reviewer" (asynchronous safety net checking PR logic and typing). 

---

## Series: "From Training to Edge in a Weekend — Agentic ML Development"

### Post 1: Training & Publishing an HF-Native Audio ML Model with Agentic Development
**Core narrative:** How structured AI-agent collaboration (not just autocomplete) shipped a complete ML pipeline from zero to published HF model.

**Key sections:**
1.  **Intro:** Brief recap of the first crack detection problem (link to POC Part 1 for background). What's different now: HF-native, edge-ready, built with a structured agentic workflow.
2.  **The Agentic Development Setup:** * `AGENTS.md` as the single source of truth — architecture map, quick commands, platform notes, all in one file the agent reads before every task.
    * Epic state management (`docs/state/registry.md` → epic file → GitHub issue) — how the agent always knows what's done and what's next.
    * Skills (`.claude/skills/`) — repeatable workflows (train-model, evaluate-model, export-onnx, push-to-hub) the agent can invoke.
    * Rules — enforcing branch naming (`feature/{issue}-{slug}`), PR conventions, co-author attribution, coding standards (ruff, pyright).
    * One PR per story, branch protection on main.
3.  **The Build:** Walk through the initial phases at a high level (not code-heavy — link to repo for details).
    * *Phase 1-2:* Scaffold → data pipeline → training (95.6% val accuracy, 208 training samples).
    * *Phase 3:* Inference with sliding window + pop-confirmation logic.
    * *Phase 4:* Publishing to HF Hub (model, model card, quickstart notebook).
4.  **Where the Agent Struggled (And How the System Caught It):** Highlight moments where Oz hallucinated or made errors (e.g., library imports, syntax) and show how the rigid project structure, rules, and tests caught it.
5.  **Highlight:** GitHub Copilot as PR reviewer — how it caught real issues across PRs #17, #20, #23 (3 rounds of fixes on #23 alone).
6.  **Results:** 91.1% test accuracy / 0.913 F1 from 298 chunks (6 roasts), model live on HF Hub.
* **Dev.to angle:** Emphasize the Warp ambassador perspective, the developer experience, how this differs from "just using ChatGPT". Use Warp Block Sharing links instead of screenshots where possible.
* **HF angle:** Emphasize the HF Trainer API, `save_pretrained`/`from_pretrained` pattern, model card best practices, quickstart notebook.

### Post 2: Dataset Expansion, Hyperparameter Tuning & Detection Latency
**Core narrative:** How re-annotating 15 recordings with a simpler approach, building a proper chunking pipeline, and tuning hyperparameters produced a model that detects first crack in <1 second on known mics.

**Key sections:**
1. **The annotation problem:** Prototype used 20-30 small 3-5s regions per file → variable-length chunks → zero-padded to 10s during training → training/inference mismatch.
2. **The fix:** Single-region annotation (one `first_crack` region per roast) + sliding window chunker (`chunk_audio.py`) → consistent 10s real-audio chunks.
3. **Recording-level splitting (Visualized):** Include a clear diagram/table showing "Data Leakage (The Wrong Way)" (overlapping chunks in train/val) vs. "Recording Split (The Right Way)" (Roast A fully in Train, Roast B fully in Val). 
4. **Hyperparameter tuning story:**
   * Attempt 1 (lr=5e-5): oscillating loss, peaks at epoch 7 then overfits. Diagnosis: lr too high for 587 training samples.
   * Attempt 2 (lr=2e-5, weight_decay 0.01→0.05, early_stop 5→3): stable training, early stops at epoch 5.
   * *The ML Lesson:* Explain *why* this worked. A massive pre-trained Transformer on a tiny dataset will suffer from catastrophic forgetting with high learning rates; higher weight decay forces generalization.
5. **Results comparison:**
   * v1: 91.1% test acc / 0.913 F1 / 3 FP (298 chunks, 6 roasts)
   * v2: **97.4% test acc / 0.925 F1 / 0 FP** (973 fixed 10s chunks, 15 roasts)
   * Precision: 87.5% → **100%**. 
   * *The Engineering Caveat:* Acknowledge that while 0 FP on the test set is excellent, real-world deployment over hundreds of roasts will inevitably encounter out-of-distribution sounds that could trigger false positives.
6. **Detection latency on full recordings:**
   * mic-1: 0.0s and 0.3s delay — near instant.
   * mic-2: 27.4s delay — model less confident on new microphone, first crack starts sparse.
   * **The Probability Timeline Chart:** Visualize this delay by plotting Model Confidence (0.0 to 1.0) against the Threshold Line (0.90) over time, showing exactly when the model crosses the threshold vs human hearing.
7. **What the agent did:** Oz created the plan, built all 3 data_prep scripts, wrote 22 tests, updated docs, ran training, ran evaluation, ran full-recording inference — all in one conversation.
* **Dev.to angle:** The hyperparameter tuning workflow with an AI agent — how it diagnosed the oscillation, proposed the fix, and validated.
* **HF angle:** The `WeightedLossTrainer` handling 20/80 class imbalance, recording-level splitting best practice, sliding window chunking for audio.

### Post 3: Deploying an Audio ML Model to Raspberry Pi 5 — The Edge Reality Check
**Core narrative:** Taking our highly-tuned cloud model to a $60 ARM board, validated entirely via SSH from Warp.

**Key sections:**
1.  **The SSH Workflow:** How Warp/Oz connected to RPi5 over SSH, ran benchmarks, debugged hardware issues — all from the same agentic conversation.
    * Agent created ONNX-only evaluation scripts (no PyTorch inference dependency on Pi).
    * Agent interpreted throttle flags (`0x50000`, `0xe0000`), diagnosed PSU and thermal issues.
    * Agent iterated on thread-limiting strategy based on live Pi results.
2.  **ONNX Export & Quantization:**
    * FP32 (345MB) → INT8 dynamic quantization (90MB) — zero quality loss.
    * Published both variants to HF Hub.
3.  **The Hardware Story (engaging content):**
    * Under-voltage crashes with standard USB-C chargers (even 96W Apple).
    * Thermal throttling without active cooling: 47°C → 77°C.
    * Thread-limiting as a survival strategy.
    * The $12 PSU fix.
4.  **Benchmarks (real numbers from the repo):**
    * Quality: 93.3% accuracy identical across Mac/Pi, FP32/INT8.
    * Latency breakdown: feature extraction 49ms (2%) vs ONNX inference 2,019ms (98%).
    * INT8 is free: 2x faster, 4x smaller, zero quality loss.
    * Production config: 2 threads, threshold=0.90, 2 cores free for MCP+UI.
    * The 10x gap: M-series Mac → ARM64 reality (197ms vs 2,452ms).
5.  **When to give up on edge:** AST at 87M params can't hit <500ms on Pi — options for the future (smaller models like MobileNet, C++ feature extraction).
6.  **Highlight:** Copilot review on edge code: How PR #23 went through 3 rounds of Copilot review fixes.
* **Dev.to angle:** The SSH story, hardware debugging narrative, agentic development on remote hardware.
* **HF angle:** ONNX export with `optimum`, publishing ONNX variants to Hub, `hf_hub_download()` for edge devices.

### Post 4 (Optional/Future): "One MCP to Rule Them All" — Preview
* A shorter post or section at the end of Post 3 that previews the future roadmap:
* Single unified MCP server (not 2 like the POC) combining: FC detection, machine monitoring, machine control, statistics capture.
* Agentic UI and orchestration.
* Why one MCP was the lesson learned from the POC.
* *Note: Explicitly out of scope for this weekend — just a teaser.*

---

## Content Checklist per Post
- [ ] Include real code snippets from the repo (not synthetic examples).
- [ ] Link to HF model: [https://huggingface.co/syamaner/coffee-first-crack-detection](https://huggingface.co/syamaner/coffee-first-crack-detection)
- [ ] Link to GitHub repo: [https://github.com/syamaner/coffee-first-crack-detection](https://github.com/syamaner/coffee-first-crack-detection)
- [ ] Link to POC series for background context.
- [ ] Reference specific PRs and issues for credibility.
- [ ] Include actual metrics tables from `docs/state/epics/` with hyperlinks to commit hashes.
- [ ] Mention Warp ambassador program (Dev.to versions).
- [ ] Cross-link between Dev.to and HF versions.
- [ ] YAML frontmatter for HF version (`title`, `thumbnail`, `authors: [{user: syamaner}]`).
- [ ] Compressed images/diagrams for both platforms.

---

## Screenshots & Assets to Capture
Before writing, scroll through Warp conversation history for the `coffee-first-crack-detection` project and capture these moments:

* **Warp Block Links (Preferred over screenshots for Dev.to):** Generate shareable links for the SSH benchmark runs, test outputs, and skill invocations.
* **`AGENTS.md` read:** Agent reading the project rules at the start of a task.
* **Epic state update:** Agent checking off a story in the epic doc after completing it.
* **SSH to RPi5:** The remote session showing benchmark output, throttle flag debugging, thermal readings.
* **Copilot PR review:** Review comments on a PR (e.g. #17 or #23) and the subsequent fix commits.
* **Data Leakage Diagram:** Flowchart showing chunk vs recording split.
* **Probability Timeline Chart:** Graphing confidence vs threshold. 

---

## Analysis Needed Before Writing
*Do this research when starting each post, not upfront:*

### For Post 1
* Review PR #16 (Phase 2&3) and PR #17 (Phase 3&4) Copilot review comments — what did Copilot catch? What categories of issues?
* Review PR #18 (training validation) — the metrics story.
* Count total Copilot review comments across all PRs and categorise them.
* Review `git log` to map the timeline.

### For Post 2
* Extract data splitting statistics (how many chunks before/after the refactor).
* Review the training logs to confirm exact epoch numbers for the oscillation vs stable runs.
* Pull the exact inference latency numbers for mic-1 and mic-2 to build the timeline chart.

### For Post 3
* Review PR #23 in detail — 3 rounds of Copilot fixes, what changed each round.
* Extract the SSH session flow: what commands were run on the Pi, in what order.
* Map the hardware debugging timeline: when was the PSU issue discovered, thermal throttling identified, thread-limiting evolved.
* Review `scripts/evaluate_onnx.py` and `scripts/benchmark_onnx_pi.py` for code snippets to include.

### For All Posts
* Review the GitHub issue #1 epic structure — how stories were organised.
* Review `AGENTS.md` evolution — did it change across PRs?
* Review `docs/state/` files to trace how state was maintained across sessions.

---

## Execution Order
1. **Posts 1 and 2 can be written in parallel** — they cover independent topics (initial setup vs data/tuning). 
2. **Post 3 should be written last**, as it relies on the finalized model from Post 2.
3. Within each post, the Dev.to and HF versions can also be prepared in parallel (shared outline, platform-specific adjustments).

