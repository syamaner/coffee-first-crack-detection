# Blog Strategy: Coffee First Crack Detection — Agentic HF Development Journey

## Context
* **What happened:** Built a complete HF-native audio ML pipeline — from scaffold to published model to RPi5 edge validation and live Gradio UI — in a single weekend using Warp/Oz.
* **The Data Contribution:** Created, annotated, and open-sourced the **first-ever public audio dataset** for coffee roasting first crack detection, filling a complete void in the open-source ML community.
* **Existing POC series:** 3 dev.to posts covering an earlier prototype (no HF integration, no Pi deployment, 2 rough MCPs). The new series **supersedes** the prototype posts — each old post should get a banner linking to the new series as the production-quality version.
* **What's new:** HF-native model, published on Hub, ONNX INT8 edge deployment on RPi5, structured agentic dev workflow (`AGENTS.md`, epics, skills, rules, SSH), Copilot PR reviews.
* **Themes:** Spec-driven development, the Senior ML Director vs. AI Coder dynamic, data leakage/re-engineering, edge hardware reality checks, and platform serving.
* **Ambassador:** Warp's first ambassador cohort.

## Publishing & Partitioning Strategy

**Dev.to Strategy: The 5-Part "Agentic Journey"**
* **Tone:** Senior Engineering Lead / Warp Ambassador.
* **Focus:** The *Developer Experience (DX)*. How Warp, Oz, Copilot, and the human interacted. The hardware debugging, the spec-driven architecture, and the reality of edge deployment.

**Hugging Face Strategy: The 2-Part "Ecosystem Masterclass"**
* **Tone:** ML Researcher / Practitioner.
* **Focus:** The *Machine Learning (ML)*. Dropping most of the "Warp/Agent" narrative to focus strictly on how to use the HF Ecosystem (Transformers, Datasets, Optimum, Gradio) to solve a hard audio problem.
* **HF Post 1:** *Training a Zero-False-Positive Audio Transformer* (Combines Dev.to Posts 1, 2 & 3).
  * Sections: problem statement, AST architecture choice, dataset creation & annotation, data pipeline (chunking → recording-level split), class imbalance + WeightedLossTrainer, AudioSet calibration transfer, hyperparameter tuning story, evaluation results with data provenance links.
* **HF Post 2:** *Deploying an 86M Parameter Audio Transformer: From Edge to Gradio* (Combines Dev.to Posts 4 & 5).
  * Sections: ONNX export via Optimum, INT8 quantization with zero quality loss, RPi5 deployment (thread-limiting, threshold tuning), Gradio Space (lazy loading, example clips, path security), model card + dataset card publishing, CI/CD with GitHub Actions.

* **Data Provenance Rule:** Every metric referenced in the series (accuracy, latency, F1) must include a hyperlink directly to the specific Git commit hash or evaluation JSON file in the repo.

---

## Dev.to Series: "From Prototype to Production in a Weekend"

### Post 1: The Architecture & The Agent (Spec-Driven ML)
**Core narrative:** You had a messy prototype. You wanted a production pipeline. You wrote the spec, handed it to Warp/Oz, and acted as the Director while the AI acted as the Senior Dev.

**Key sections:**
1. **The Hook (front-load the payoff):** Open with the finished product — live Gradio Space link, 97.4% accuracy / 100% precision stat, photo of the RPi5 running inference. Show what was built *before* explaining how. "Here is the production pipeline we shipped in 25 hours. Now let me show you the exact system that made it possible."
2. **The Dataset Tension (tease Post 2):** Immediately after the hook, plant the stake: "We didn't just wrap an API. There was no public dataset for coffee roasting audio anywhere on the internet — not HuggingFace, not Kaggle, not in any paper. We had to build one from scratch, navigate audio data leakage, and handle 20/80 class imbalances. (The full data story is in Post 2.)" Two sentences, maximum — creates anticipation without stealing Post 2's narrative.
3. **The Prototype → Production Bridge:** Brief link-back to the existing 3-part dev.to prototype series. Frame it as: "here's the end-to-end idea we proved; this series is the production-quality iteration."
4. **The "Director/Coder" Dynamic:** Explicitly define the boundaries: **Human** sets the architecture and reviews ML science; **Warp/Oz** executes the terminal, writes boilerplate, and invokes skills; **Copilot** acts as the async safety net.
5. **The Agentic Setup — as a steal-this-template:** `AGENTS.md` as the single source of truth, Epic state management (`registry.md`), and parameterized skills (`.claude/skills/`). Include a **generalised, copy-pasteable `AGENTS.md` boilerplate** stripped of coffee-specific details — readers can drop it into their own projects. This makes the post useful independent of the coffee domain and dramatically increases bookmark/share rate on dev.to.
6. **The Build & The Fails:** Walk through the scaffold to initial training. Highlight where Oz hallucinated (library imports, syntax) and how the strict project rules caught it. Include a concrete before/after code diff showing what Oz generated vs. what the human corrected — grounds an otherwise process-heavy post in real code.
7. **Copilot Reality Check:** Show how Copilot caught 28 batches of typing/API errors, but *never* caught a logic/ML error.
8. **The Numbers Sidebar:** Quick-reference box: 18 stories, 10 PRs, ~55 commits (52 with Oz co-author), 25 hours wall-clock. Makes the post shareable for the developer productivity audience.

### Post 2: The Data — Building the First Public Coffee Roasting Audio Dataset
**Core narrative:** No public audio dataset for coffee roasting first crack existed anywhere — not on HuggingFace, not on Kaggle, not in academic papers. Before we could train anything, we had to build the dataset from scratch. This post is about that contribution and the data engineering that makes or breaks audio ML.

**Key sections:**
1. **The Gap:** Search HuggingFace Datasets, Kaggle, Papers with Code — nothing. The open-source ML community has zero labelled audio for coffee roasting. Frame the dataset release as the primary open-source contribution of this project.
2. **Recording & Annotation:** The physical setup (microphones, roasting sessions), Label Studio annotation workflow, and the evolution from fragmented multi-region annotations to the single-region continuous approach. Why the annotation methodology matters as much as the model.
3. **Data Leakage (Visualized):** Why chunk-level splitting ruins audio models. Diagram showing "The Wrong Way" (overlapping chunks from the same recording in train + test) vs. "The Right Way" (recording-level split). This is the section that will resonate most with ML practitioners.
4. **Class Imbalance Handling:** The 20/80 imbalance (first_crack vs no_first_crack) and the `WeightedLossTrainer` with class-weighted CrossEntropyLoss. Why naive training under-detects the minority class.
5. **The Dataset by the Numbers:** 973 chunks, 15 roasts (9 mic-1, 6 mic-2), 197 first_crack / 776 no_first_crack, recording-level 70/15/15 split. Link to the published HF Dataset Card.

### Post 3: The Science — Hyperparameter Tuning & Model Evolution
**Core narrative:** AI can write training loops, but it can't fix learning rate schedules or diagnose catastrophic forgetting. This post is about the human ML engineering required to go from 87.5% precision to 100%.

**Key sections:**
1. **Transfer Learning Subtlety:** Using AudioSet calibration values (`mean=-4.2677, std=4.5689`) for `ASTFeatureExtractor` rather than computing dataset-specific stats. Why this matters for a pre-trained AST model.
2. **The Tuning Story:**
   * Attempt 1 (lr=5e-5): Oscillating loss, overfits by epoch 7. Diagnosis: LR too high for 587 samples.
   * Attempt 2 (lr=2e-5, higher weight decay): Stable training. The lesson on catastrophic forgetting.
3. **baseline_v1 → baseline_v2 Transition:** What drove the retraining — the annotation methodology change (fragmented → continuous sliding window) *and* the mic-2 data expansion (6 → 15 roasts). The metrics shift: from 3 FP to 0 FP, but recall dropped (95.5% → 86.1%). Explain the precision/recall tradeoff as a deliberate design choice for roasting automation.
4. **The Results:** 97.4% accuracy, 100% precision, 0.982 ROC-AUC. Adding the Senior ML caveat: 0 FP on a test set is great, but real-world out-of-distribution sounds will still challenge it.
5. **Detection Latency on Full Recordings:** The sliding-window results — 0.0s and 0.3s delay on mic-1, but 27.4s on mic-2. Honest analysis of why mic-2 lags (less training data from that mic type).

### Post 4: The Edge — Porting to Raspberry Pi 5
**Core narrative:** Hardware reality checks. Taking a heavy cloud model (AST 86M parameters) and forcing it onto a $60 ARM board using SSH straight from Warp.

**Key sections:**
1. **What We're Deploying (model recap for edge-first readers):** Brief 3-4 sentence recap: AST-based binary classifier, 86M parameters, trained on 973 chunks from 15 roasts, 97.4% test accuracy / 100% precision. Links to Posts 2 & 3 for the full training story. This makes the post self-contained for RPi/edge readers who land here directly.
2. **The SSH Workflow:** How Oz connected to the RPi5, ran benchmarks, and debugged hardware issues remotely.
3. **ONNX Export:** FP32 (345MB) → INT8 dynamic quantization (90MB) with zero quality loss.
4. **Metrics Clarification:** Compare v2 ONNX INT8 accuracy against v2 PyTorch baseline (96.86% vs 97.4% on Mac — update with Pi numbers once experiments complete). Emphasise that INT8 quantization preserves quality (0.54% gap, 1 additional FP). If citing v1 ONNX numbers (93.3%) for historical context, make it unmistakable that the gap was from different test sets, not quantization.
5. **Threshold Sweep:** The sweep from 0.50–0.95, choosing 0.90 for Pi deployment. Why minimising false positives matters more than raw accuracy for roasting automation.
6. **The Hardware War Story:**
   * Under-voltage crashes with standard USB-C chargers (including Apple 96W).
   * Diagnosing under-voltage (`vcgencmd get_throttled` → `0x50000`) and thermal throttling (`0xe0000`).
   * The multi-agent debugging: How Warp/Oz, Gemini, and the human each contributed.
   * The fix: official 27W (5V/5A) PSU + active cooler. Before: crashes at >1 thread, 77°C without fan. After: stable 4-thread operation at 45°C.
7. **The Verdict:** Achieving 2.07s latency (4 threads, with fan) / 2.45s (2 threads, recommended). Perfect for coffee roasting, but Transformers remain heavy for edge AI.

### Post 5: From Model to Product (Serving the Model)
**Core narrative:** A model isn't finished until users can touch it. Getting a model a UI in 2026, and the platform realities of Hugging Face Hub.

**Key sections:**
1. **The Packaging & Open Source Milestone:** Pushing the model, generating the Model Card, and officially releasing the Dataset Card. Emphasize that the dataset release is a first-of-its-kind asset given to the HF community.
2. **The Widget Illusion:** The assumption that `pipeline_tag: audio-classification` magically creates an inference widget. The reality check: HF requires commercial providers for custom audio pipelines.
3. **The Pivot to Gradio:** Building a Space as the correct community-serving path.
4. **Agentic Debugging in the Container:** Collapse platform compatibility bugs (YAML validation, `HfFolder` deprecation, Gradio 6.x path security) into a summary table showing bug → symptom → fix. Then expand only the SSR + Python 3.13 asyncio `ValueError` (#32) as the detailed war story — it's the most interesting and the one where Oz reading live container logs matters most.
   * *The Point:* Oz navigated all four by reading live container logs, not local tests.
5. **CI/CD & Next Steps:** GitHub Actions workflow for automated HF Hub sync (#34). Shows pipeline maturity beyond "it works on my machine".
6. **The Final Result:** Live Space at huggingface.co/spaces/syamaner/coffee-first-crack-detection.

---

## Content Checklist per Post
- [ ] Include real code snippets from the repo (not synthetic examples).
- [ ] Link to HF model: syamaner/coffee-first-crack-detection
- [ ] Link to GitHub repo: syamaner/coffee-first-crack-detection
- [ ] Link to live Gradio Space.
- [ ] Reference specific PRs and issues for credibility.
- [ ] Warp Block Links (preferred over screenshots for Dev.to).
- [ ] Cross-link between Dev.to and HF versions.
- [ ] YAML frontmatter for HF versions (title, thumbnail, authors).
- [ ] Dev.to tags (max 4 per post). Suggested pool: `#machinelearning`, `#python`, `#audio`, `#edgecomputing`, `#huggingface`, `#raspberrypi`, `#ai`, `#webdev`.

---

## Post File Structure

All posts live under `docs/posts/`. Each platform gets its own subdirectory:

```
docs/posts/
  plan.md                 ← this file
  devto/
    post-1-architecture.md
    post-2-data.md
    post-3-science.md
    post-4-edge.md
    post-5-serving.md
  hf/
    post-1-training.md
    post-2-deployment.md
  assets/
    diagrams/             ← data leakage diagram, probability timeline, threshold sweep
    screenshots/          ← Warp Block Links preferred, but fallback PNGs here
```

---

## Screenshots & Assets to Capture
Before writing, scroll through Warp conversation history and capture:

* **AGENTS.md read:** Agent reading project rules at the start of a task.
* **TODO list + skill invocation:** Agent creating a task list and invoking a skill.
* **Epic state update:** Agent checking off a story in the epic doc after completing it.
* **SSH to RPi5:** The remote session showing benchmark output, throttle flag debugging, thermal readings.
* **Copilot PR review:** Review comments on PR #23 and the subsequent fix commits.
* **Data Leakage Diagram:** Flowchart showing chunk vs recording split.
* **Probability Timeline Chart:** Graphing confidence vs threshold over a full recording.
* **The "not deployed" screenshot:** Model card showing "This model isn't deployed by any Inference Provider".
* **RPi5 hardware debugging:** `vcgencmd get_throttled` output showing `0x50000` and `0xe0000` — the before/after with PSU + cooler fix.
* **Threshold sweep chart:** ROC-AUC curve or threshold vs F1/precision/recall chart from the Pi sweep data.
* **Generalised AGENTS.md template:** Prepare a copy-pasteable boilerplate stripped of coffee-specific details for Post 1's steal-this-template section.

---

## Analysis Needed Before Writing

### For Posts 1 & 2 (Architecture + Data)
* Pull all Copilot comments per PR from GitHub API. Categorise: type safety, error handling, API misuse, docs/copy, dependency management.
* Lock down the "28 review batches" stat — verify from GitHub API before writing.
* Verify exact commit timestamps per phase from GitHub API to support the timeline narrative.
* Check which commits carry the Oz co-author tag.
* Run `git diff --stat FIRST_COMMIT HEAD` for total lines added/removed.

### For Post 3 (Science / Tuning)
* Extract exact training logs for Attempt 1 vs Attempt 2 — loss curves, epoch-by-epoch metrics.
* Verify baseline_v1 → baseline_v2 metric deltas against evaluation JSONs in experiments/.
* Prepare detection latency data for the full-recording sliding window results.

### For Post 4 (Edge / RPi5)
* **Run all baseline_v2 cross-platform experiments** (see experiment matrix above) — this must complete before writing Post 4.
* Review PR #23 in detail — 3 rounds of Copilot fixes, what changed each round.
* Extract the SSH session flow: commands run on the Pi in order.
* Map the hardware debugging timeline: PSU issue, thermal throttling, thread-limiting evolution.
* Document the multi-agent debugging: which steps were Warp/Oz, Gemini, or manual.

### For Post 5 (Serving)
* Verify exact Gradio SDK version history to accurately describe the HfFolder removal timeline.
* Screenshot the live Space with a classification result showing.
* Document the SSR bug (#32) — reproduce the exact error trace and the one-line fix.
* Capture the GitHub Actions sync workflow design (#34) for the CI/CD section.

---

## Epic Analysis & Stats (For Post Fact-Checking)

### Timeline Reality
* **Total wall-clock time:** ~25.5 hours (2026-04-03 22:06 to 2026-04-04 23:36). Note: post-epic housekeeping commits extend to 2026-04-05 — scope the "weekend" framing to the code/ML work only.
* **The Narrative:** "Built in a single weekend." Spans two calendar days and includes overnight sleep.
* **Git Stats (verify at write time):** ~65 total commits, ~55 non-merge, 10 PRs merged, 18 stories completed across 6 phases. 52 commits carry the Oz `Co-Authored-By` tag.
* **Verify:** "28 review batches" Copilot stat from GitHub API — must be locked down before writing.

### PR Breakdown
| PR | Content | Opened | Merged |
|----|---------|--------|--------|
| #16 | Train, eval, inference (Phase 2 & 3) | 2026-04-03 | 2026-04-03 |
| #17 | Export, scripts, tests (Phase 3 & 4) | 2026-04-03 | 2026-04-03 |
| #18 | Training validation (S7) | 2026-04-03 | 2026-04-03 |
| #19 | Epic state docs | 2026-04-03 | 2026-04-03 |
| #20 | Quickstart notebook (S13) | 2026-04-03 | 2026-04-04 |
| #21 | Model card update (S12) | 2026-04-04 | 2026-04-04 |
| #23 | RPi5 ONNX validation (S15) | 2026-04-04 | 2026-04-04 |
| #27 | Data prep + mic-2 expansion (S16/S17) | 2026-04-04 | 2026-04-04 |
| #28 | Gradio Space + widget (S18) | 2026-04-04 | 2026-04-04 |
| #29 | Epic complete state docs | 2026-04-04 | 2026-04-04 |

### Key Model Results

**baseline_v1 (PyTorch, 45-sample test set from 6 roasts):**
|| Model | Test Acc | F1 | Precision (FC) | Recall (FC) | FP |
||-------|----------|----|----------------|-------------|----|
|| baseline_v1 PyTorch (MPS) | 91.1% | 0.913 | 87.5% | 95.5% | 3 |
|| v1 ONNX FP32 (Mac) | 93.3% | 0.933 | 91.3% | 95.5% | 2 |
|| v1 ONNX INT8 (Mac) | 93.3% | 0.933 | 91.3% | 95.5% | 2 |
|| v1 ONNX FP32 (RPi5) | 93.3% | 0.933 | 91.3% | 95.5% | 2 |
|| v1 ONNX INT8 (RPi5) | 93.3% | 0.933 | 91.3% | 95.5% | 2 |

**baseline_v2 (191-sample test set from 15 roasts, re-annotated):**
|| Model | Test Acc | F1 | Precision (FC) | Recall (FC) | FP | Status |
||-------|----------|----|----------------|-------------|----|---------|
|| baseline_v2 PyTorch (MPS) | 97.4% | 0.925 | 100% | 86.1% | 0 | ✅ done |
|| v2 ONNX INT8 (Mac) | 96.86% | 0.912 | 96.9% | 86.1% | 1 | ✅ done (`results/v2_mac_int8_eval.json`) |
|| v2 ONNX FP32 (Mac) | — | — | — | — | — | ⬜ TODO |
|| v2 ONNX INT8 (RPi5, 4T) | 96.86% | 0.912 | 96.9% | 86.1% | 1 | ✅ done (`results/v2_pi5_int8_4t_eval.json`) |
|| v2 ONNX FP32 (RPi5) | — | — | — | — | — | ⬜ TODO |
|| v2 PyTorch (CUDA) | — | — | — | — | — | ⬜ TODO |
|| v2 ONNX INT8 (CUDA) | — | — | — | — | — | ⬜ TODO |
|| v2 ONNX FP32 (CUDA) | — | — | — | — | — | ⬜ TODO |

**Key observation:** v2 ONNX INT8 on Mac (96.86%) closely matches v2 PyTorch (97.4%) — the 0.54% gap and 1 additional FP is minimal. This confirms INT8 quantization preserves quality on v2, unlike the misleading v1-vs-v2 comparison that showed a 4% gap due to different test sets.

---

### Baseline v2 Cross-Platform Experimentation

Before writing Posts 3-4, run the full baseline_v2 comparison across all target platforms. This gives the blog series a complete, honest cross-platform benchmark table.

**Experiment matrix:**

*MacBook (Apple Silicon):*
- PyTorch MPS: ✅ done (97.4%, training evaluation)
- ONNX FP32 CPU: ⬜ run `evaluate_onnx.py` with v2 FP32 export
- ONNX INT8 CPU: ✅ done (96.86%, `results/v2_mac_int8_eval.json`)

*Raspberry Pi 5 (ARM64, ONNX Runtime only):*
- ONNX FP32 CPU (2 threads): ⬜ run `evaluate_onnx.py` via SSH
- ONNX INT8 CPU (2 threads): ⬜ run `evaluate_onnx.py` via SSH
- ONNX INT8 CPU (4 threads, fan): ✅ done (`results/v2_pi5_int8_4t_eval.json`)
- Latency benchmark (FP32 + INT8): ⬜ run `benchmark_onnx_pi.py` via SSH

*NVIDIA CUDA (RTX 4090 or similar):*
- PyTorch CUDA (fp16/bf16): ⬜ run `benchmark_platforms.py` with `--model-dir`
- ONNX CUDAExecutionProvider FP32: ⬜ needs `benchmark_platforms.py` update to support CUDA EP
- ONNX CUDAExecutionProvider INT8: ⬜ needs `benchmark_platforms.py` update to support CUDA EP

**Infrastructure changes needed:**
1. **Re-export ONNX from baseline_v2 checkpoint** — ensure `exports/onnx/` contains v2 models
2. **Extend `benchmark_platforms.py`** — add `CUDAExecutionProvider` support for ONNX (currently CPU-only)
3. **Run `evaluate_onnx.py` on all platforms** with v2 test set (191 samples) for consistent accuracy comparison
4. **Save all results** to `results/` with `v2_` prefix naming convention

**Naming convention for result files:**
- `results/v2_mac_fp32_eval.json`
- `results/v2_mac_int8_eval.json` (already exists)
- `results/v2_pi5_fp32_2t_eval.json`
- `results/v2_pi5_int8_2t_eval.json`
- `results/v2_pi5_int8_4t_eval.json`
- `results/v2_cuda_pytorch_eval.json`
- `results/v2_cuda_fp32_eval.json`
- `results/v2_cuda_int8_eval.json`
- `results/v2_latency_benchmark.json` (all platforms combined)

**How results feed into posts:**
- **Post 3 (Science):** v2 PyTorch MPS vs baseline_v1 comparison shows the impact of re-annotation + data expansion
- **Post 4 (Edge):** Full cross-platform table (Mac vs Pi vs CUDA × FP32 vs INT8) with latency — the centrepiece of the post
- **HF Post 1:** v2 accuracy table with data provenance links
- **HF Post 2:** Complete latency + quality table across all platforms

### Detection Latency (full-length recordings, baseline_v2)
| Recording | Mic | FC Onset | Detected | Delay |
|-----------|-----|----------|----------|-------|
| 25-10-19_1236-brazil-3 | mic-1 | 452.7s | 453.0s | 0.3s |
| mic2-brazil-roast2 | mic-2 | 599.6s | 627.0s | 27.4s |
| roast-2-costarica-hermosa-hp-a | mic-1 | 441.0s | 441.0s | 0.0s |

### Copilot Review Impact
* 28 review batches across 10 PRs.
* PR #23 (RPi5 validation): 3+ rounds of review fixes.
* PR #27 (data prep): validation checks and error handling.
* PR #28 (Gradio Space): caught misleading UI copy, missing explicit dependency.
* Conclusion: Copilot excels at code hygiene and API misuse. Architecture remained human+Oz.

---

## Warp Ambassador Strategy

### Why This Series Hits Differently
Most ambassador content falls into the "Hello World" trap. This series bypasses that entirely: it shows a Senior ML Engineer using Warp to ship a multi-modal, hardware-integrated pipeline, and open-source a novel dataset, all in 25 hours. 

### The Four Angles to Lean Into

**1. Warp's Unique Value Prop: Terminal + AI**
Oz is *in the terminal* — not in a browser tab. Spell this out:
- Oz reads the Gradio container logs when it crashes, and fixes it.
- Oz SSHes into the RPi5, runs `vcgencmd`, reads hex codes, and adjusts ONNX threads.
- Explicit line in Post 4 (Edge): "The amount of context-switching time saved because the AI lives inside the terminal."

**2. "Director vs. Doer" is the Future of Coding**
Warp's vision: developers architect, the AI executes boilerplate.
- You wrote `AGENTS.md` (the spec). Oz did the typing.
- Every post should open with the human decision that drove it, then show the agent executing it.

**3. Warp Block Sharing Over Screenshots**
Replace static screenshots with **Warp Block Links**:
- RPi5 SSH session with vcgencmd output.
- Gradio container crash log + fix.
- When readers click a Block Link, they land in Warp's web UI — highly organic product placement.

**4. Authenticity Builds Trust**
Include the failures:
- Attempt 1 hyperparameter oscillation.
- The 4 Gradio container bugs.
- Copilot catching 28 batches of errors.
- The mic-2 27.4s detection delay.

### Launch Tactics
1. **Tag founders/DevRel on publish** — Frame it as a deep dive into *Agentic Development with Oz*. 
2. **Warp Discord** — Drop it in community channels highlighting the `AGENTS.md` context-loading pattern.
3. **Cross-post sequence** — Post 1 → wait 2-3 days → Post 2 → 2-3 days → etc. Tighter cadence preserves momentum on dev.to; series with >2 week gaps lose readers. Don't drop all five at once.
4. **The AGENTS.md trick as a standalone thread** — Explain the `AGENTS.md` → epic registry → GitHub issue pattern on Twitter/LinkedIn.
5. **The Dataset Drop hook** — Use the "I couldn't find a public dataset, so I built the first one using Warp" hook heavily on LinkedIn and Twitter.
6. **LinkedIn strategy** — Write a standalone LinkedIn post per Dev.to article highlighting the "Director vs Doer" methodology and concrete metrics (25h, 18 stories, 100% precision).