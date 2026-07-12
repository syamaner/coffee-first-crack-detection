# RF + MFCC comparator — pre-registered protocol

Written and committed to disk **before** the training/eval script has been run against the
test set. This file's content is not edited after seeing test-set results; findings go in
a separate `RESULTS.md` written after the run.

## Purpose

A classical, non-transformer comparator for the AST-based first-crack detector, per the
Shi et al. 2026 reference point (Applied Food Research 6(1):102058 — RF on MFCC features,
~95.7% accuracy / 0.992 AUC on their own data, not directly comparable since it's a
different dataset). This is a same-data head-to-head against our AST int8 model, not an
attempt to reproduce their paper's numbers.

## Data

- **Train**: `data/splits/train/` (922 chunks: 136 `first_crack` / 786 `no_first_crack`,
  21 recordings' worth minus val/test, recording-level split, seed=42 — same split the AST
  model trains on).
- **Test**: `data/splits/test/` (303 chunks: 42 `first_crack` / 261 `no_first_crack`) — the
  same 303-sample set used for every other comparator in this repo. Used ONLY for final
  evaluation, never for feature selection or hyperparameter choice.
- `data/splits/val/` is available but not used — see hyperparameter policy below (no tuning
  loop that would need it).
- No data augmentation (the repo's amplitude-scaling/noise augmentation is AST-training
  specific and not applied here — RF gets the raw chunked audio, mirroring what the
  comparator paper would plausibly do with a fixed feature set).

## Features

Standard MFCCs via `librosa.feature.mfcc`, not the Kaldi-compatible `MelFrontend` the AST
torch-free path needed — this is a comparator baseline, not a from-scratch numerical
equivalence target, so librosa's standard implementation is the appropriate and simpler
choice per the task brief.

- Sample rate: 16000 Hz (matches `SAMPLE_RATE` used throughout the repo)
- `n_mfcc=20` (a conventional default for speech/audio classification; not tuned)
- `n_fft=2048`, `hop_length=512` (librosa defaults)
- Per-clip feature vector: mean + standard deviation of each MFCC coefficient across time
  frames (a standard fixed-length summarisation for a fixed-length classifier input) → 40
  features per 10s clip (20 mean + 20 std)
- No delta / delta-delta MFCCs, no additional spectral features — keeping the feature set
  minimal and standard, consistent with "does not need the Kaldi-fbank equivalence" framing
  in the task brief

## Model

- `sklearn.ensemble.RandomForestClassifier`
- **Hyperparameter policy: default sklearn RF parameters, no tuning.** The repo has no
  existing RF-tuning convention to follow, and the task brief specifies "default sklearn RF
  unless the repo has a tuning convention" — it doesn't, so defaults it is
  (`n_estimators=100`, `criterion="gini"`, no max_depth cap, etc., i.e. whatever
  `RandomForestClassifier()` gives out of the box in the pinned scikit-learn version).
- One exception: `random_state=42` (repo seed convention, `configs/default.yaml`) and
  `class_weight="balanced"` — the training split is imbalanced (136:786, ~15% positive) and
  leaving RF unweighted on a 6:1 imbalance would degenerate to a majority-class-biased
  classifier without this being a form of tuning against the test set; `balanced` is a
  standard, data-blind rebalancing scheme (inverse of class frequency), not a
  hyperparameter searched against results.
- No cross-validation / grid search — a single fit on the full training split, as specified
  by "default sklearn RF" in the task brief.

## Metrics (reported for both models on the test set)

Accuracy, precision, recall, F1 (all for the `first_crack` positive class, matching the
convention `evaluate.py` / `evaluate_onnx.py` already use in this repo), ROC-AUC, and
per-window inference latency (feature extraction + predict, end-to-end, matching how
`evaluate_onnx.py` times ONNX latency).

## Comparator

AST INT8 (`exports/onnx/int8`), using the already-measured 12 Jul numbers from the #55
reconciliation (`results/baseline_v5_303set/onnx_int8_eval.json`) — not re-run, since
nothing about the AST model changes here; re-quoting the existing measurement avoids a
redundant multi-minute ONNX pass.

## What counts as "favourable to RF"

Any metric where RF's test-set number is equal to or higher than AST-int8's. Reported
honestly regardless of outcome — this document is written before the comparison is run, so
there is no result yet to be favourable or unfavourable to.
