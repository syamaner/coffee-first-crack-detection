# RF + MFCC comparator — results (12 Jul)

Protocol pre-registered in `PROTOCOL.md` in this directory, written and committed before this
script was run against the test set. Nothing in the protocol was changed after seeing these
numbers.

## Head-to-head vs AST INT8

| Metric | RF + MFCC | AST INT8 (ONNX) | Favours |
|---|---|---|---|
| Accuracy | 86.80% | 98.35% | AST |
| Precision (`first_crack`) | 52.94% | 91.11% | AST |
| Recall (`first_crack`) | 42.86% | 97.62% | AST |
| F1 | 0.474 | 0.943 | AST |
| ROC-AUC | 0.904 | 0.998 | AST |
| Confusion (TN/FP/FN/TP) | 245/16/24/18 | 257/4/1/41 | AST |
| Inference latency (per 10s window, this Mac) | ~7.6ms p50 | ~216ms p50 | **RF** |

AST is favoured on every quality metric; RF wins only on latency, by roughly 28×.

**AST INT8 numbers** are quoted from the 12 Jul #55 reconciliation
(`results/baseline_v5_303set/onnx_int8_eval.json`, quality; `latency_benchmark_mac.json` /
`scripts/benchmark_platforms.py`, latency) rather than re-run here — nothing about the AST
model changes in this comparison, so re-running would be redundant.

**Latency methodology note**: the two latency figures share the same measurement *boundary*
(isolated feature-extraction + model-inference call, no file I/O — neither includes the
`librosa.load()` disk read) but not the same *sampling scheme*, so "~7.6ms vs ~216ms" is not
an apples-to-apples repeat-count comparison:
- AST INT8 (216ms p50): `benchmark_platforms.py`'s controlled benchmark — **one fixed
  synthetic 10s window, timed 30 times** after 5 warmup runs, p50 taken across those 30
  repeats of the same input.
- RF+MFCC (~7.6ms p50): this script — **303 distinct real test-set windows, each timed
  once**, p50 taken across those 303 different inputs (no repeat timing of a single window,
  no separate warmup).
Both are legitimate "warm, isolated inference call" measurements (as opposed to the slower
end-to-end `evaluate_onnx.py` eval-loop timing, which also includes file I/O and runs ~637ms
p50 for the same INT8 model on this machine) — but the RF figure additionally reflects
real across-sample variance (different audio content, different feature-extraction cost per
clip) that the AST figure's fixed-input repeat sampling does not. The ~28× gap is real and
not an artifact of this difference (RF's `p95` of 7.72ms in the bare-default run is still
~28× below AST's own warmup-adjusted range), but the two numbers were not produced by
identical harnesses.

## Honest read

RF on plain-vanilla MFCC summary statistics is not competitive with the fine-tuned AST model
on this dataset — recall in particular collapses to 42.9% (24 missed first-cracks out of 42),
which is disqualifying for this application (a missed FC over-roasts until an operator
override — see AGENTS.md's stated precision/recall priority). The 28× latency advantage does
not offset a >2x drop in recall.

This is consistent with the task's expectation that RF+MFCC is a comparator baseline, not a
contender: Shi et al. 2026 report ~95.7% accuracy / 0.992 AUC for RF+MFCC on their own
(different) dataset — this repo's 21-recording, two-microphone, real-roast dataset with
mic-gain variation across recordings (see README Limitations) is evidently harder for a
fixed, non-learned feature representation than whatever produced their result; the AST
model's advantage here plausibly comes from the pretrained AudioSet representation (72M
frozen params) plus fine-tuning, which a 40-dimensional MFCC summary can't match. No claim is
made about why Shi et al.'s number was higher — different data, different preprocessing,
different label definition are all plausible and unverified from here.

**Not favourable to RF on any accuracy-family metric.** Reporting this as such per the task
brief's explicit instruction to call out honestly if the comparison favours RF on any metric —
it does not, except latency.

**Transparency check on `class_weight="balanced"`**: since this is the one non-bare-default
choice in the protocol, it was checked against a truly bare `RandomForestClassifier
(random_state=42)` (no `class_weight`, i.e. `--class-weight none`) to confirm it wasn't
inadvertently cherry-picked toward a flattering number. This check is reproducible from the
committed CLI, not a one-off interactive run — see `--class-weight` below. Bare-default
result (`results/rf_mfcc_baseline/RESULTS_bare_default.json`): 85.8% accuracy, 48.8%
precision, 50.0% recall, F1 0.494, ROC-AUC 0.879, confusion 239/22/21/21. That's *higher* F1
than the balanced run (0.494 vs 0.474) and lower ROC-AUC/accuracy — a wash, not an
improvement, confirming `class_weight="balanced"` was not selected because it flattered RF.
Both configurations land far short of AST on every quality metric; the headline RF numbers in
the table above are the protocol-specified `class_weight="balanced"` run.

## Reproduce

```bash
# Protocol default (class_weight="balanced")
python scripts/rf_mfcc_baseline.py \
  --train-dir data/splits/train --test-dir data/splits/test \
  --output results/rf_mfcc_baseline/RESULTS.json

# Transparency check (bare sklearn default, no class_weight)
python scripts/rf_mfcc_baseline.py \
  --train-dir data/splits/train --test-dir data/splits/test \
  --class-weight none \
  --output results/rf_mfcc_baseline/RESULTS_bare_default.json
```

Raw JSON: `results/rf_mfcc_baseline/RESULTS.json` (protocol default),
`results/rf_mfcc_baseline/RESULTS_bare_default.json` (transparency check).
