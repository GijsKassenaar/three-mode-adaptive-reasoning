# Reproducing comparison and 7B scale-up

This document is the runbook for additions on top of the 1.5B MATH paper
runs: **prior-method comparisons** (AdaptThink, AutoThink) and a **single 7B
seed**. All reuse `scripts/paper_protocol.sh`. Do not change those constants in
one job and not the others.

Data is not committed. Generate it once:

```bash
bash scripts/prepare_math_lighteval_math500.sh
bash scripts/prepare_benchmarks.sh
```

That writes `data/math_lighteval/{train,dev}.parquet` (train = MATH-lighteval,
dev = MATH-500) and `data/{gsm8k,aime2024,aime2025}/`.

---

## Shared protocol (must stay aligned)

| Knob | Value | Where it is used |
|---|---|---|
| Prompt / response cap | 2048 / 16384 | train + eval |
| Train batch / group size G | 128 / 8 | train |
| Actor LR | 1e-6 | train |
| Loss aggregation | token-mean | train |
| KL in reward / std-normalized GRPO | off | train |
| Mode caps | 1024 / 3000 / 16384 | train (eval uncapped) |
| Bases / γ (phase 1) | 1.3, 1.2, 1.0 and 0.99984 / 0.99994 / 1.0 | train steps 0–60 |
| Phase 2 | all bases 1.0, all γ = 1.0 | train steps 61–90 |
| Warmup | 45 steps, `G//4` forced per mode | train |
| Val temperature / top_p | 0.6 / 1.0 | train val + all eval jobs |
| GSM8K / MATH-500 / AIME | avg@5 / avg@5 / avg@16 | benchmark eval |

Training validation still uses `n=1` (same as the 1.5B job). Benchmark tables
use the n above, same as `scripts/eval_benchmarks_all.job`.

---

## 1. AdaptThink comparison

We **evaluate released HF weights** under this paper’s eval, we do not retrain
AdaptThink. Their training data is DeepScaleR; ours is MATH-lighteval. The
comparison is therefore a **method + checkpoint** comparison at matched
inference settings, not a matched-data reimplementation.

Routing is **off**. AdaptThink chooses Think vs NoThink inside its own policy.
Injecting `NOTHINK/SHORT/LONG` would not be their method.

```bash
# Main 1.5B AdaptThink checkpoint (δ = 0.05)
MODEL=THU-KEG/AdaptThink-1.5B-delta0.05 TAG=adaptthink_1p5b_d005 \
  sbatch scripts/eval_hf_no_routing.job

# Optional extra δ points (same protocol)
MODEL=THU-KEG/AdaptThink-1.5B-delta0 TAG=adaptthink_1p5b_d000 \
  sbatch scripts/eval_hf_no_routing.job
MODEL=THU-KEG/AdaptThink-1.5B-delta0.1 TAG=adaptthink_1p5b_d010 \
  sbatch scripts/eval_hf_no_routing.job

# AutoThink Stage 3 (ellipsis hybrid, routing off, same job)
MODEL=SONGJUNTU/Distill-R1-1.5B-AutoThink-Stage3 TAG=autothink_1p5b_s3 \
  sbatch scripts/eval_hf_no_routing.job

# Untrained 1.5B base on the same no-routing protocol
# (scripts/eval_benchmarks_baseline.job is the same setting)
sbatch scripts/eval_benchmarks_baseline.job
```

Report accuracy **and** mean response length on the same three benchmarks as
Figure 6 / Figure 7. Each eval job prints both at the end of the log:

- accuracy: `val-core/.../acc/mean@N`
- length: `val-aux/response_length/mean` (also `val-lengths/response/mean`)

Place AdaptThink on that acc-vs-length plane next to free routing, Long-only,
Short-only, NoThink-only, and the untrained base.

Our trained router is still evaluated with
`VAL_MODE=FREE sbatch scripts/eval_benchmarks_all.job <ckpt>/global_step_90`
(routing **on**; `FREE` is the job default). Do not mix that job with
`eval_hf_no_routing.job` when filling one table row.

### Measured (Snellius, 2026-08-30)

`THU-KEG/AdaptThink-1.5B-delta0.05`, `TAG=adaptthink_1p5b_d005`, routing off,
paper protocol (temp 0.6, top_p 1.0, max 16384, boxed scorer). FSDP used
SDPA (`+actor_rollout_ref.model.override_config.attn_implementation=sdpa`);
standalone flash-attn 2.8.3 was not used because it broke vLLM.

Logs on the cluster checkout: `eval_bench_{aime,math500,gsm8k}_adaptthink_1p5b_d005.log`.

MATH-500 L1–L5 acc (AdaptThink only): 0.921 / 0.833 / 0.874 / 0.784 / 0.639.
AIME 12.1% truncated at 16k.

### Measured AutoThink (Snellius, 2026-08-31)

`SONGJUNTU/Distill-R1-1.5B-AutoThink-Stage3`, `TAG=autothink_1p5b_s3`, routing
off, same protocol and SDPA FSDP as AdaptThink. Their tokenizer chat template
already contains the ellipsis (`<think>\n...\n`); we do not inject
`NOTHINK/SHORT/LONG`. They train on DeepScaleR-style data; this is again a
**method + checkpoint** comparison. Their original eval uses top_p=0.95 and
32k context; we keep top_p=1.0 and 16k.

Logs: `eval_bench_{aime,math500,gsm8k}_autothink_1p5b_s3.log`.

MATH-500 L1–L5 acc: 0.907 / 0.876 / 0.882 / 0.811 / 0.693.
AIME 11.3% truncated at 16k. AIME length is one pooled mean over 2024+2025.

### Full comparison (Table 5 protocol)

Base and free-routing numbers are paper Table 5. AdaptThink and AutoThink are
the two HF evals above.

| | MATH-500 | GSM8K | AIME’24 / ’25 | Length MATH / GSM8K / AIME |
|---|---|---|---|---|
| Base | 0.796 | 0.831 | 0.265 / 0.231 | 4743 / 1930 / 12266 |
| Free routing (three-mode) | 0.782 | 0.781 | 0.276 / 0.222 | 2810 / 459 / 10716 |
| AdaptThink-1.5B δ=0.05 | 0.785 | 0.829 | 0.281 / 0.219 | 1464 / 638 / 7212 |
| AutoThink-1.5B Stage3 | 0.814 | 0.843 | 0.367 / 0.260 | 2093 / 659 / 8112 |

- **Acc:** AutoThink is the strongest of the three methods on all three
  suites (above Table 5 base on MATH, GSM8K, and AIME). AdaptThink matches
  base on GSM8K and is ~Table 5 on MATH/AIME. Three-mode free routing pays a
  GSM8K acc cost (0.781 vs 0.831).
- **Length:** AdaptThink is shortest on MATH and AIME. Three-mode is
  shortest on GSM8K (459). AutoThink sits between AdaptThink and three-mode
  on MATH/AIME and near AdaptThink on GSM8K.
- **Caveat:** AdaptThink and AutoThink were not trained on MATH-lighteval.
  AIME length is one pooled mean, not two year-specific length stats.

---

## 2. One 7B seed

Same two-phase GRPO job as 1.5B, model
`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, seed **2**.

```bash
sbatch scripts/train_three_mode_routing_math_7b.job
```

Overrides (only if the 4×H100 job OOMs; keep algorithm knobs unchanged):

```bash
N_GPUS=8 ROLLOUT_TP_SIZE=2 GPU_MEM_UTIL=0.35 \
  sbatch scripts/train_three_mode_routing_math_7b.job
```

After `global_step_90`:

```bash
# Free routing, paper benchmark protocol (VAL_MODE defaults to FREE)
BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  ROLLOUT_TP_SIZE=2 GPU_MEM_UTIL=0.4 \
  sbatch scripts/eval_benchmarks_all.job \
  /scratch-shared/$USER/three_mode_routing_ckpts/three_mode_routing_math_7b_seed2/global_step_90

# Untrained 7B base (no routing)
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B TAG=base_7b \
  ROLLOUT_TP_SIZE=2 GPU_MEM_UTIL=0.4 \
  sbatch scripts/eval_hf_no_routing.job
```

A 7B run counts as a scale check if, at step 90:

1. all three free-rollout mode fractions are non-zero,
2. routing entropy is not collapsed,
3. either per-mode accuracy inverts (brief modes > Long) **or** the MATH-500
   level split still moves mass from NoThink on easy levels to Long on hard ones.

---

## What this does not cover

- Collapse ablation (caps off, `balance_coef=0`) — not in this branch.
- Retraining AdaptThink or AutoThink on MATH-lighteval.
- ARM / Thinkless (supervised-stage methods).
