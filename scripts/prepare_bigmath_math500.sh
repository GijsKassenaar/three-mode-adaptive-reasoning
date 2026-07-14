#!/usr/bin/env bash
set -euo pipefail

# Prepare the Big-Math-RL-Verified train split with MATH-500 (from MATH-lighteval test)
# as the dev split. Test parquet is the remainder of the MATH-lighteval test split.
#
# Big-Math is a GATED dataset on HuggingFace. Before running:
#   1. Accept terms at https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified
#   2. Export HF_TOKEN=<your token>  (or run `huggingface-cli login`)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/data/bigmath"}
HDFS_DIR=${HDFS_DIR:-""}
NUM_PROC=${NUM_PROC:-8}

mkdir -p "$OUTPUT_DIR"

ARGS=("--local_dir" "$OUTPUT_DIR" "--num_proc" "$NUM_PROC")
if [[ -n "$HDFS_DIR" ]]; then
  ARGS+=("--hdfs_dir" "$HDFS_DIR")
fi

python examples/data_preprocess/bigmath.py "${ARGS[@]}"

printf "\nWrote:\n  %s\n  %s\n  %s\n" \
  "$OUTPUT_DIR/train.parquet" \
  "$OUTPUT_DIR/dev.parquet" \
  "$OUTPUT_DIR/test.parquet"
