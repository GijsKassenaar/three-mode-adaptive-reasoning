"""Build GSM8K and AIME validation parquets in the same format as the MATH-lighteval
sets, so the routing model can be evaluated on them with no other changes.

Both use the boxed prompt (matching how the routed model answers) and are scored by
the boxed `math_reward` scorer via the "gsm8k" / "aime2024" / "aime2025" data_source
mappings in verl/utils/reward_score/__init__.py.

Writes  data/gsm8k/{train,dev}.parquet  and  data/aime/{train,dev}.parquet
(`dev` is the eval split; `train` is written only so the eval job's dataloader builds —
for AIME it duplicates the 90 problems since AIME has no train split).
"""
import argparse
import os

import datasets

INSTR = "Let's think step by step and output the final answer within \\boxed{}."


def _row(question, answer, data_source, split, idx):
    return {
        "data_source": data_source,
        "prompt": [{"content": f"{question} {INSTR}", "role": "user"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer).strip()},
        "extra_info": {"index": idx, "split": split, "source": data_source},
    }


def build_gsm8k(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for split, hf_split, fname in [("train", "train", "train.parquet"), ("test", "test", "dev.parquet")]:
        ds = datasets.load_dataset("openai/gsm8k", "main", split=hf_split)

        def fn(ex, idx):
            # GSM8K gold answer is the text after "#### "
            ans = ex["answer"].split("####")[-1].strip().replace(",", "")
            return _row(ex["question"], ans, "gsm8k", split, idx)

        ds = ds.map(fn, with_indices=True, remove_columns=ds.column_names)
        ds.to_parquet(os.path.join(out_dir, fname))
        print(f"  gsm8k {fname}: {len(ds)} rows")


# Single-year AIME sets (30 Q each) so numbers are comparable to published AIME24/AIME25.
AIME_YEARS = {
    "aime2024": ("Maxwell-Jia/AIME_2024", "train", "Problem", "Answer", "aime2024"),
    "aime2025": ("yentinglin/aime_2025", "train", "problem", "answer", "aime2025"),
}


def build_aime_year(key, out_dir):
    hf_id, hf_split, qcol, acol, ds_name = AIME_YEARS[key]
    os.makedirs(out_dir, exist_ok=True)
    ds = datasets.load_dataset(hf_id, split=hf_split)

    def fn(ex, idx):
        return _row(ex[qcol], ex[acol], ds_name, "test", idx)

    ds = ds.map(fn, with_indices=True, remove_columns=ds.column_names)
    # AIME has no train split; write the same set as dev (eval) and train (dataloader build).
    ds.to_parquet(os.path.join(out_dir, "dev.parquet"))
    ds.to_parquet(os.path.join(out_dir, "train.parquet"))
    print(f"  {key} dev/train: {len(ds)} rows  (data_source={ds_name})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.path.join(os.getcwd(), "data"))
    ap.add_argument("--which", default="all", choices=["all", "gsm8k", "aime2024", "aime2025"])
    args = ap.parse_args()

    if args.which in ("all", "gsm8k"):
        print("Building GSM8K...")
        build_gsm8k(os.path.join(args.data_root, "gsm8k"))
    for key in ("aime2024", "aime2025"):
        if args.which in ("all", key):
            print(f"Building {key}...")
            build_aime_year(key, os.path.join(args.data_root, key))
    print("done")
