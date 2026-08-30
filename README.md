# Adaptive Three-Mode Routing for GRPO

Reinforcement-learning post-training that teaches a reasoning LLM to **choose its own
inference budget per problem**. The model emits one of three routing words as the first
token of every response:

| Mode | First token | Intended behaviour | Reward (correct) |
|---|---|---|---|
| NOTHINK | `NOTHINK` | answer with little or no reasoning | `nothink_base × γ_nothink^L` |
| SHORT | `SHORT` | brief reasoning | `short_base × γ_short^L` |
| LONG | `LONG` | extended reasoning | `long_base` (no discount) |
| — invalid | anything else | — | `unknown_penalty` |

with `L` = generated tokens after the routing word (or reasoning-only tokens with
`reasoning_only=True`). Routing is learned **implicitly through this shaped reward
surface plus per-mode hard token caps — no auxiliary loss, no extra data**: a routing
question is injected into every prompt at generation time, and during a warmup phase a
fraction of each prompt's GRPO group is *forced* into each mode so the routing words
enter the response head and receive gradient.

The result is a difficulty-calibrated router: easy problems route to NOTHINK/SHORT for
large token savings, hard problems keep the full LONG budget. Trained on
MATH-lighteval with a DeepSeek-R1-Distill-Qwen-1.5B base, the router transfers
zero-shot to GSM8K (~3× fewer tokens at near-equal accuracy) and AIME (accuracy
preserved), routing ~99% of GSM8K to NOTHINK and ~99% of AIME to LONG.

This repository is [verl](https://github.com/volcengine/verl) **v0.7.0 with exactly one
method built on top**. Everything specific to this work is listed in
[CHANGES_VS_VERL.md](CHANGES_VS_VERL.md); the core implementation is a single module,
[verl/trainer/ppo/three_mode_routing.py](verl/trainer/ppo/three_mode_routing.py).

---

## Method components

1. **Routing question + forced prefixes** (`RoutingPromptController`). The question
   *"Output NOTHINK to answer directly, SHORT for brief reasoning, or LONG for extended
   reasoning:"* is appended to the last user message at runtime. For forced rollouts the
   routing word is additionally placed after the chat template's assistant prefix, then
   *promoted* from the prompt tail into the response head after generation so it is
   trained like any generated token.

2. **Phased rollout protocol** (`ThreeModeRoutingForcer`). With group size
   `n_rollouts = G`: phase 1 (steps ≤ `warmup_steps`) forces `G//4` rollouts per mode and
   leaves the rest free; optional phase 2 (≤ `warmup2_steps`) forces 1 per mode; after
   warmup all rollouts are free (the model routes itself).

3. **Per-mode hard token caps** (`apply_three_mode_caps`). The primary mode separator:
   NOTHINK/SHORT responses are truncated at their caps *before scoring*, so an answer
   beyond the cap is wrong by construction. Forced rollouts also generate with
   `max_tokens` set to their cap (no wasted tokens). Reward shaping alone cannot
   separate modes on a 1.5B model — the caps make the modes mechanically distinct.

4. **Advantage estimator** (`compute_three_mode_routing_advantage`, registered as
   `algorithm.adv_estimator=three_mode_routing`). Shaped rewards are mean-centered per
   group (std-normalization disabled — reward magnitudes are intentionally
   heterogeneous), with unknown rollouts excluded from the mean so a large
   `unknown_penalty` cannot make wrong-but-valid groupmates positive.

5. **Load-balance term.** `balance_coef × (balance_target − frac_mode)` added at the
   advantage level over **free** rollouts only, optionally correctness-gated
   (`balance_correctness_gated`) and never penalizing wrong LONG attempts
   (`balance_protect_long_incorrect`). A linear anneal schedule releases the split
   after routing is established.

All knobs live under `algorithm.three_mode_routing.*` — see the fully commented block in
[verl/trainer/config/ppo_trainer.yaml](verl/trainer/config/ppo_trainer.yaml).

---

## Setup

The code requires the same environment as verl v0.7.0 (PyTorch, Ray, vLLM ≥ 0.8, FSDP).
On a fresh machine:

```bash
git clone <this-repo> three-mode-routing && cd three-mode-routing
pip install -e .            # or follow verl's install docs (README_verl.md)
```

The SLURM scripts in `scripts/` assume the Snellius layout (partition `gpu_h100`,
4×H100 per node, conda env `zero`, scratch under `/scratch-shared/$USER`); adapt the
`#SBATCH` headers, `module load`, and `conda activate` lines for other clusters. All
scripts honour `REPO_DIR` (repo checkout), `DATA_DIR`, `BASE_MODEL`, and
`EXPERIMENT_NAME` environment overrides.

### Data

```bash
bash scripts/prepare_math_lighteval_math500.sh   # MATH-lighteval train + MATH-500 dev
bash scripts/prepare_benchmarks.sh               # GSM8K, AIME 2024, AIME 2025 val sets
```

No routing-specific preprocessing exists — the routing question is injected at
generation time from the standard parquets.

---

## Training

### Main run

```bash
sbatch scripts/train_three_mode_routing_math.job
```

Fresh start from `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`;

### Ablations

| Script | What it isolates |
|---|---|
| `scripts/train_forced_single_mode_math.job` (`MODE=NOTHINK\|SHORT\|LONG`) | per-mode floor/ceiling: every rollout forced into one mode for the whole run, validation forced into the same mode |
| `scripts/train_three_mode_family_seeds.job` | SLURM array: {forced-NOTHINK, forced-SHORT, forced-LONG, adaptive} × seeds |

### Evaluation

```bash
# MATH-500 with every prompt forced into one mode (or FREE routing):
VAL_MODE=LONG CHECKPOINT=<ckpt_dir>/global_step_90 sbatch scripts/eval_forced_routing_math.job

# GSM8K (avg@5), MATH-500 (avg@5), AIME'24+'25 (avg@16) on one checkpoint:
sbatch scripts/eval_benchmarks_all.job <ckpt_dir>/global_step_90
sbatch scripts/eval_benchmarks_baseline.job      # untrained base model reference
```

### AdaptThink comparison and 7B scale-up

See [docs/experiments.md](docs/experiments.md) for the matched protocol, jobs, and
what to report. Shared constants live in `scripts/paper_protocol.sh`.

```bash
# Released AdaptThink-1.5B (δ=0.05), no routing question, paper eval protocol:
MODEL=THU-KEG/AdaptThink-1.5B-delta0.05 TAG=adaptthink_1p5b_d005 \
  sbatch scripts/eval_hf_no_routing.job

# One 7B seed, same two-phase recipe as the 1.5B MATH run:
sbatch scripts/train_three_mode_routing_math_7b.job
```

The per-difficulty routing-split figure is produced from any training/eval log with:

```bash
python scripts/plot_routing_split_by_difficulty.py "my run=three_mode_routing_math_phase2.log"
```

---

## Monitoring (W&B)

Everything specific to the method is logged under `three_mode_routing/` (training) and
`val-aux/three_mode_routing/` (validation):

| Metric | What to watch |
|---|---|
| `{nothink,short,long,unknown}_fraction` | mode split over **free** rollouts; unknown → 0 after warmup |
| `routing_entropy` | H over free-mode fractions; max ln 3 ≈ 1.099, collapse → 0 |
| `{mode}_correct_rate` | per-mode accuracy — NOTHINK/SHORT > LONG once routing is difficulty-calibrated |
| `{mode}_mean_length` | is the cap binding? |
| `{mode}_mean_advantage` | systematic bias check |
| `{mode}_truncated_fraction` | fraction of rollouts clipped by the mode cap |
| `frac_mixed_groups`, `mean_unique_modes_per_group` | within-group mode diversity of free rollouts |
| `balance_coef_effective`, `gate_active`, `n_forced_per_mode`, `n_free` | schedule state |

Validation additionally logs response/reasoning length distributions
(`val-lengths/…`), truncation fractions, and per-MATH-level accuracy + routing split
(`val-difficulties/level{1..5}/…`) whenever the data carries a `level` field.

---

## Tests

CPU unit tests for the reward shaping, group normalization, balance term, mode
detection, and caps:

```bash
pytest tests/trainer/ppo/test_three_mode_routing_on_cpu.py -q
```

## Relation to verl

Base: [verl](https://github.com/volcengine/verl) v0.7.0 (Apache 2.0 — see LICENSE;
upstream README preserved as [README_verl.md](README_verl.md)). Every deviation from
upstream is documented in [CHANGES_VS_VERL.md](CHANGES_VS_VERL.md).
