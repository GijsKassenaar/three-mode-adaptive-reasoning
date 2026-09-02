#!/usr/bin/env python3
"""Parse committed eval_bench_*.log files into the related-work table cells.

Usage:
    python3 scripts/parse_eval_bench_logs.py
    python3 scripts/parse_eval_bench_logs.py --logs-dir docs/results/logs
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ACC = re.compile(r"val-core/(\S+)/acc/mean@(\d+):([0-9.]+)")
LENGTH = re.compile(r"val-aux/response_length/mean:([0-9.]+)")
TRUNC = re.compile(r"val-aux/truncated_by_length/fraction:([0-9.]+)")
GEN_SEED = re.compile(r"'seed': (None|\d+)")
DATA_SEED = re.compile(r"'data_loader_seed': (\d+)")
DO_SAMPLE = re.compile(r"'do_sample': (True|False)")
TEMP = re.compile(r"'temperature': ([0-9.]+)")

TABLE_ROUND_ACC = 3
TABLE_ROUND_LEN = 0

RUNS = {
    "adaptthink_1p5b_d005": {
        "display": "AdaptThink-1.5B δ=0.05",
        "checkpoint": "THU-KEG/AdaptThink-1.5B-delta0.05",
    },
    "autothink_1p5b_s3": {
        "display": "AutoThink-1.5B Stage3",
        "checkpoint": "SONGJUNTU/Distill-R1-1.5B-AutoThink-Stage3",
    },
}


def last(matches):
    return matches[-1] if matches else None


def parse_log(path: Path) -> dict:
    text = ANSI.sub("", path.read_text(errors="replace"))
    accs = ACC.findall(text)
    length = last(LENGTH.findall(text))
    trunc = last(TRUNC.findall(text))
    gen_seeds = GEN_SEED.findall(text)
    # Hydra dumps many seeds; generation seed for vLLM is the last 'seed' that is None.
    gen_seed = last([s for s in gen_seeds if s == "None"] or gen_seeds)
    return {
        "log": str(path),
        "acc": {name: {"n": int(n), "mean": float(v)} for name, n, v in accs},
        "response_length_mean": float(length) if length else None,
        "truncated_by_length_fraction": float(trunc) if trunc else None,
        "generation_seed": None if gen_seed in (None, "None") else int(gen_seed),
        "data_loader_seed": int(last(DATA_SEED.findall(text)) or 0) or None,
        "do_sample": last(DO_SAMPLE.findall(text)) == "True",
        "temperature": float(last(TEMP.findall(text)) or "nan"),
    }


def table_row(tag: str, logs_dir: Path) -> dict:
    math500 = parse_log(logs_dir / f"eval_bench_math500_{tag}.log")
    gsm8k = parse_log(logs_dir / f"eval_bench_gsm8k_{tag}.log")
    aime = parse_log(logs_dir / f"eval_bench_aime_{tag}.log")
    math_name = next(iter(math500["acc"]))
    math_acc = math500["acc"][math_name]["mean"]
    gsm_acc = gsm8k["acc"]["gsm8k"]["mean"]
    a24 = aime["acc"]["aime2024"]["mean"]
    a25 = aime["acc"]["aime2025"]["mean"]
    return {
        **RUNS[tag],
        "tag": tag,
        "table": {
            "math500_acc": round(math_acc, TABLE_ROUND_ACC),
            "gsm8k_acc": round(gsm_acc, TABLE_ROUND_ACC),
            "aime2024_acc": round(a24, TABLE_ROUND_ACC),
            "aime2025_acc": round(a25, TABLE_ROUND_ACC),
            "length_math": int(round(math500["response_length_mean"], TABLE_ROUND_LEN)),
            "length_gsm8k": int(round(gsm8k["response_length_mean"], TABLE_ROUND_LEN)),
            "length_aime_pooled": int(round(aime["response_length_mean"], TABLE_ROUND_LEN)),
        },
        "raw": {
            "math500": math500,
            "gsm8k": gsm8k,
            "aime": aime,
        },
        "notes": {
            "aime_length": "one pooled mean over aime2024+aime2025; per-year acc is in raw.aime.acc",
            "generation_seed": "vLLM generation seed is None (do_sample=True, temp 0.6). AIME n=30 is noisy.",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", type=Path, default=Path("docs/results/logs"))
    ap.add_argument("--out", type=Path, default=Path("docs/results/related_work.json"))
    args = ap.parse_args()
    payload = {
        "protocol": "scripts/paper_protocol.sh via scripts/eval_hf_no_routing.job",
        "routing": "off",
        "runs": {tag: table_row(tag, args.logs_dir) for tag in RUNS},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")
    for tag, row in payload["runs"].items():
        t = row["table"]
        print(
            f"{tag}: MATH {t['math500_acc']} GSM8K {t['gsm8k_acc']} "
            f"AIME {t['aime2024_acc']}/{t['aime2025_acc']} "
            f"len {t['length_math']}/{t['length_gsm8k']}/{t['length_aime_pooled']}"
        )


if __name__ == "__main__":
    main()
