#!/usr/bin/env bash
# rebuild_and_train.sh — Full data-prep + training pipeline.
#
# Re-chunks all annotated recordings, splits into train/val/test,
# trains the model, and runs test-set evaluation.
#
# Usage:
#   ./scripts/rebuild_and_train.sh <experiment-name>
#
# Example:
#   ./scripts/rebuild_and_train.sh baseline_v6
#
# Prerequisites:
#   - venv activated (source venv/bin/activate)
#   - Per-file annotation JSONs in data/labels/
#   - Corresponding WAV files in data/raw/
#
# Co-Authored-By: Oz <oz-agent@warp.dev>

set -euo pipefail

EXPERIMENT_NAME="${1:?Usage: $0 <experiment-name>}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT"

# Verify venv is active and resolve its Python binary
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "❌ Virtual environment not active. Run: source venv/bin/activate"
  exit 1
fi

# Use venv Python explicitly — shell aliases (e.g. python -> homebrew)
# don't work in scripts.
PYTHON="${VIRTUAL_ENV}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "❌ venv Python not found at ${PYTHON}"
  exit 1
fi

START_TIME=$SECONDS

echo "=============================================="
echo "  Coffee First Crack — Rebuild & Train"
echo "  Experiment: ${EXPERIMENT_NAME}"
echo "=============================================="

# ── Step 1: Clean existing processed + splits ────────────────────────────────
echo ""
echo "📦 Step 1/5: Cleaning existing processed data and splits..."
rm -rf data/processed/first_crack data/processed/no_first_crack data/processed/processing_summary.md
rm -rf data/splits/train data/splits/val data/splits/test data/splits/split_report.md
echo "   Done."

# ── Step 2: Chunk all annotated recordings ───────────────────────────────────
echo ""
echo "🎵 Step 2/5: Chunking all annotated recordings into 10s windows..."
"$PYTHON" -m coffee_first_crack.data_prep.chunk_audio \
  --labels-dir data/labels \
  --audio-dir data/raw \
  --output-dir data/processed

# ── Step 3: Stratified train/val/test split ──────────────────────────────────
echo ""
echo "🔀 Step 3/5: Splitting into train/val/test (70/15/15, recording-level)..."
"$PYTHON" -m coffee_first_crack.data_prep.dataset_splitter \
  --input data/processed \
  --output data/splits \
  --train 0.7 --val 0.15 --test 0.15 \
  --seed 42

# ── Step 4: Train ────────────────────────────────────────────────────────────
echo ""
echo "🏋️ Step 4/5: Training model (experiment: ${EXPERIMENT_NAME})..."
"$PYTHON" -m coffee_first_crack.train \
  --data-dir data/splits \
  --experiment-name "${EXPERIMENT_NAME}"

# ── Step 5: Evaluate on test set ─────────────────────────────────────────────
echo ""
echo "📊 Step 5/5: Evaluating on test set..."
"$PYTHON" -m coffee_first_crack.evaluate \
  --model-dir "experiments/${EXPERIMENT_NAME}/checkpoint-best" \
  --test-dir data/splits/test \
  --output-dir "experiments/${EXPERIMENT_NAME}/evaluation"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Pipeline complete!"
echo "  Checkpoint: experiments/${EXPERIMENT_NAME}/checkpoint-best"
echo "  Evaluation: experiments/${EXPERIMENT_NAME}/evaluation/"
echo "  Split report: data/splits/split_report.md"
echo "  Duration: $(( (SECONDS - START_TIME) / 60 ))m $(( (SECONDS - START_TIME) % 60 ))s"
echo "=============================================="
