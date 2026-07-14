#!/usr/bin/env bash
set -euo pipefail

# Prepare the MATH-lighteval train split and the MATH-500 dev split (from test indices).
# Writes  data/math_lighteval/{train,dev,test}.parquet:
#   train.parquet — MATH-lighteval train split (7500 problems)
#   dev.parquet   — MATH-500 (the standard 500-problem eval subset of the test split)
#   test.parquet  — remainder of the test split (excluded from training and dev)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/data/math_lighteval"}
HDFS_DIR=${HDFS_DIR:-""}

mkdir -p "$OUTPUT_DIR"

ARGS=("--local_dir" "$OUTPUT_DIR")
if [[ -n "$HDFS_DIR" ]]; then
  ARGS+=("--hdfs_dir" "$HDFS_DIR")
fi

python examples/data_preprocess/math_lighteval.py "${ARGS[@]}"

printf "\nWrote:\n  %s\n  %s\n  %s\n" \
  "$OUTPUT_DIR/train.parquet" \
  "$OUTPUT_DIR/dev.parquet" \
  "$OUTPUT_DIR/test.parquet"
