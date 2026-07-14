# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Preprocess SynthLabsAI/Big-Math-RL-Verified (~250K RL-curated, verifiable math
problems decontaminated against MATH-500/AIME-2024/AMC-2023/OlympiadBench/MinervaMath/GSM8K).
Dev/test are taken from DigitalLearningGmbH/MATH-lighteval using the MATH-500 indices,
matching the existing math_lighteval pipeline so validation curves remain comparable.

NOTE: Big-Math-RL-Verified is a *gated* dataset. You must first visit
https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified and accept the terms,
then have a valid HF_TOKEN in the environment (or be logged in via
`huggingface-cli login`) before running this script.
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

# Reuse the MATH-500 dev split from math_lighteval.py so validation matches existing runs.
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math_lighteval import DEV_INDICES


INSTRUCTION_FOLLOWING = "Let's think step by step and output the final answer within \\boxed{}."

TRAIN_SOURCE = "SynthLabsAI/Big-Math-RL-Verified"
EVAL_SOURCE = "DigitalLearningGmbH/MATH-lighteval"


def _strip_boxed(answer: str) -> str:
    """Big-Math `answer` is usually already a bare answer string, but some entries
    include `\\boxed{...}`. Normalize defensively."""
    if not isinstance(answer, str):
        return answer
    answer = answer.strip()
    boxed = last_boxed_only_string(answer)
    if boxed is not None:
        try:
            return remove_boxed(boxed)
        except Exception:
            return answer
    return answer


def _extract_math_solution(solution_str: str) -> str:
    return remove_boxed(last_boxed_only_string(solution_str))


def build_train(num_proc: int):
    print(f"Loading {TRAIN_SOURCE} from huggingface...", flush=True)
    ds = datasets.load_dataset(TRAIN_SOURCE, split="train", trust_remote_code=True)

    def process(example, idx):
        question = example["problem"] + " " + INSTRUCTION_FOLLOWING
        ground_truth = _strip_boxed(example["answer"])
        return {
            "data_source": TRAIN_SOURCE,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {"split": "train", "index": idx},
        }

    return ds.map(
        process,
        with_indices=True,
        num_proc=num_proc,
        remove_columns=ds.column_names,
    )


def build_dev_and_test(num_proc: int):
    print(f"Loading {EVAL_SOURCE} (for MATH-500 dev/test) from huggingface...", flush=True)
    ds = datasets.load_dataset(EVAL_SOURCE, trust_remote_code=True)
    test_dataset = ds["test"]

    def process(example, idx):
        question = example["problem"] + " " + INSTRUCTION_FOLLOWING
        ground_truth = _extract_math_solution(example["solution"])
        return {
            "data_source": EVAL_SOURCE,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {"split": "test", "index": idx},
        }

    test_dataset = test_dataset.map(
        process,
        with_indices=True,
        num_proc=num_proc,
        remove_columns=test_dataset.column_names,
    )

    dev_indices_set = set(DEV_INDICES)
    dev_dataset = test_dataset.select(DEV_INDICES)
    test_dataset = test_dataset.filter(
        lambda _example, idx: idx not in dev_indices_set,
        with_indices=True,
    )
    return dev_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/bigmath")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset = build_train(args.num_proc)
    dev_dataset, test_dataset = build_dev_and_test(args.num_proc)

    print(
        f"Sizes - train: {len(train_dataset)}  dev: {len(dev_dataset)}  test: {len(test_dataset)}",
        flush=True,
    )

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    dev_dataset.to_parquet(os.path.join(local_dir, "dev.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
