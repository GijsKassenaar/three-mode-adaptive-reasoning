#!/usr/bin/env bash
set -euo pipefail

# Prepare the cross-benchmark validation sets (GSM8K, AIME 2024, AIME 2025)
# in the same parquet format as the MATH-lighteval sets, so a routing checkpoint
# can be evaluated on them with no other changes.
# Writes  data/{gsm8k,aime2024,aime2025}/{train,dev}.parquet

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT=${DATA_ROOT:-"$REPO_ROOT/data"}

python examples/data_preprocess/extra_benchmarks.py --data_root "$DATA_ROOT" --which all
